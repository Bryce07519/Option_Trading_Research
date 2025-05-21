from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Option_strat:
    def __init__(self,path,size) -> None:
        pass
        self.Step_size=size
        
    def read_df(self,path):
        # 最简单：pandas 自动选择 engine
        df=pd.read_parquet(path)
        # 1. 读入期权表 df，包含 minute_str, Instrument, strike, iv
        df['minute'] = pd.to_datetime(df['minute_str'])
        df['T'] = df['maturity']  / 252
        df.is_call=[bool(int(each)) for each in df.is_call.values]
        return  df
    def get_underly_F(self,df):
        unified = (
            df
            .groupby(['minute', 'is_call'])['underly_mean_mid']
            .mean()
            .round()
            .astype(int)
            .rename('underly_F')
            .reset_index()
            )
        # 2. 将统一后的价格合并回原始 df
        df = df.merge(unified, on=['minute', 'is_call'], how='left')
        return df
    
    def get_call_df(self,df,call=True):
        return df[df.is_call] if call else df[~df.is_call]
    
    def bs_future_option_price(self,F, K, T, sigma, is_call=True):
        T = max(T, 1e-12)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if is_call:
            return F * norm.cdf(d1) - K * norm.cdf(d2)
        else:
            return K * norm.cdf(-d2) - F * norm.cdf(-d1)

    def implied_volatility(self,mid_price, F, K, T, is_call=True):
        obj = lambda sigma: self.bs_future_option_price(F, K, T, sigma, is_call) - mid_price
        try:
            return brentq(obj, 1e-6, 5.0)
        except ValueError:
            return np.nan
        
    def add_iv(self,call_df):
        # 2. 计算 IV
        call_df['iv'] = call_df.apply(
            lambda row: self.implied_volatility(
                mid_price = row['mean_mid'],
                F         = row['underly_mean_mid'],
                K         = row['strike'],
                T         = row['T'],
                is_call   = row['is_call']
            ),
            axis=1
        )
        return call_df
    def ave_iv(self,df):

        # 2. 按 minute_str 和 strike 分组，计算平均 iv
        iv_by_minute_strike = (
            df
            .groupby(['minute', 'strike'])['iv']
            .mean()
            .reset_index()
            .rename(columns={'iv': 'iv_mean'})
        )

        # 3. （可选）把平均 iv 合并回原始 df
        df = df.merge(iv_by_minute_strike, on=['minute', 'strike'], how='left')
        return df
    
    def get_local_skew(self,df):

        # 假设 df 已有：minute (datetime), T, strike, iv, F
        # 1. 排序 & 分组
        df = df.sort_values(['minute','T','strike']).reset_index(drop=True)

        # 2. 找 ATM：每组 strike 与 S 差值最小
        atm = (
            df.assign(diff=lambda d: (d['strike'] - d['underly_F']).abs())
            .loc[:, ['minute','T','strike','iv','diff']]
            .groupby(['minute','T'], as_index=False)
            .apply(lambda g: g.loc[g['diff'].idxmin()])
            .rename(columns={'strike':'strike_atm', 'iv':'iv_atm'})
            .drop(columns='diff')
        )

        # 3. 合并回原 df
        df = df.merge(atm, on=['minute','T'], how='left')

        # 4. 计算 strike_diff_steps 与 skew_per_step
        df['strike_diff_steps'] = (df['strike'] - df['strike_atm']) / self.Step_size
     

        df['skew_per_step'] = df['strike_diff_steps'].replace(0, np.nan).pipe(
            lambda d: (df['iv'] - df['iv_atm']) / d
        )
        df['strike_diff_steps'] = df['strike_diff_steps'].astype(int)
        return df
    def bact_test(self,df,quantile=[0.05,0.95],W=1000,MP=500):

        # 1. 只保留目标档位
        valid_steps = list(range(-5,0)) + list(range(1,14))
        df = df[df['strike_diff_steps'].isin(valid_steps)].reset_index(drop=True)

        # 2. 计算滚动分位数 & 中位数
       
        grp = df.groupby('strike_diff_steps')['skew_per_step']
        df['q10'] = grp.transform(lambda x: x.rolling(W, MP).quantile(quantile[0]))
        df['q50'] = grp.transform(lambda x: x.rolling(W, MP).quantile(0.50))
        df['q90'] = grp.transform(lambda x: x.rolling(W, MP).quantile(quantile[-1]))

        # 3. 生成信号
        df['prev_skew'] = df['skew_per_step'].shift(1)
        df['prev_q10']  = df['q10'].shift(1)
        df['prev_q90']  = df['q90'].shift(1)

        df['signal_put']  = (df['prev_skew'] <= df['prev_q90']) & (df['skew_per_step'] > df['q90'])
        df['signal_call'] = (df['prev_skew'] >= df['prev_q10']) & (df['skew_per_step'] < df['q10'])

        # 4. 建立快速取价表
        # 索引 (minute, T, strike, is_call) -> (ask, bid)
        price_table = df.set_index(['minute','T','strike','is_call'])[['mean_ask','mean_bid']]
        # 5. 遍历信号，计算 PnL
        trades = []
        for steps, group in df.groupby('strike_diff_steps'):
            g = group.reset_index(drop=True)
            # PUT-SPREAD trades
            for i in g.index[g['signal_put']]:
                t0 = g.loc[i,'minute']; T0 = g.loc[i,'T']
                K1 = g.loc[i,'strike']                              # long put at this strike
                K2 = g.loc[i,'strike_atm'] + (steps-1)*200           # short put one step lower
                # 检查两腿是否都存在当刻报价
                key1 = (t0,T0,K1,False); key2 = (t0,T0,K2,False)
                if key1 not in price_table.index or key2 not in price_table.index:
                    continue
                # 开仓成本（debit）
                ask1 = price_table.loc[key1,'mean_ask']
                bid2 = price_table.loc[key2,'mean_bid']
                debit = ask1 - bid2
                # 找到回归至中位的平仓时刻
                sub = g.loc[i+1:]
                hit = sub[sub['skew_per_step'] <= sub['q50']]
                if hit.empty: 
                    continue
                t1 = hit.iloc[0]['minute']
                # 平仓价
                key1x = (t1,T0,K1,False); key2x = (t1,T0,K2,False)
                if key1x not in price_table.index or key2x not in price_table.index:
                    continue
                bid1x = price_table.loc[key1x,'mean_bid']
                ask2x = price_table.loc[key2x,'mean_ask']
                credit = bid1x - ask2x
                pnl = credit - debit
                trades.append({'type':'put','steps':steps,'t0':t0,'t1':t1,'pnl':pnl})
        
            # CALL-SPREAD trades
            for i in g.index[g['signal_call']]:
                t0 = g.loc[i,'minute']; T0 = g.loc[i,'T']
                K2 = g.loc[i,'strike']                              # short call at this strike
                K1 = g.loc[i,'strike_atm'] + (steps-1)*200           # long call one step below
                key1 = (t0,T0,K1,True); key2 = (t0,T0,K2,True)
                if key1 not in price_table.index or key2 not in price_table.index:
                    continue
                ask1 = price_table.loc[key1,'mean_ask']
                bid2 = price_table.loc[key2,'mean_bid']
                debit = ask1 - bid2
                sub = g.loc[i+1:]
                hit = sub[sub['skew_per_step'] >= sub['q50']]
                if hit.empty:
                    continue
                t1 = hit.iloc[0]['minute']
                key1x = (t1,T0,K1,True); key2x = (t1,T0,K2,True)
                if key1x not in price_table.index or key2x not in price_table.index:
                    continue
                bid1x = price_table.loc[key1x,'mean_bid']
                ask2x = price_table.loc[key2x,'mean_ask']
                credit = bid1x - ask2x
                pnl = credit - debit
                trades.append({'type':'call','steps':steps,'t0':t0,'t1':t1,'pnl':pnl})
        # 6. 汇总结果
        trades_df = pd.DataFrame(trades)
        print("总笔数：", len(trades_df))
        print("总 PnL：", trades_df['pnl'].sum())
        print("平均单笔 PnL：", trades_df['pnl'].mean())
        print("胜率：", (trades_df['pnl']>0).mean())

        # 按方向/档位统计
        print(trades_df.groupby('type')['pnl'].agg(['count','mean','sum','std']))
        print(trades_df.groupby('steps')['pnl'].agg(['count','mean']).sort_index())
    def get_atm_iv(self,df,features=['iv_atm','minute',]):
        ATM_iv=df[features]
        ATM_iv=ATM_iv.sort_values('minute')
        ATM_IV_minute=ATM_iv.groupby('minute').mean()
        del ATM_iv
        return ATM_IV_minute

    def plot_hist(self,df):
            # 1. 整体分布：histogram
        plt.figure(figsize=(8,4))
        plt.hist(df['skew_per_step'].dropna(), bins=50)
        plt.title('Distribution of skew_per_step')
        plt.xlabel('skew_per_step')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
    def plot_box(self,df):
        # 2. 档位对比：boxplot
        buckets = sorted(df['strike_diff_steps'].unique())
        data = [df.loc[df['strike_diff_steps']==b, 'skew_per_step'].dropna()
                for b in buckets]
        plt.figure(figsize=(12,6))
        plt.boxplot(data, labels=buckets, showfliers=False)
        plt.title('Boxplot of skew_per_step by strike_diff_steps')
        plt.xlabel('strike_diff_steps')
        plt.ylabel('skew_per_step')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    def plot_mean_vol(self,df):
        # 3. 档位统计：均值±标准差
        agg = df.groupby('strike_diff_steps')['skew_per_step'].agg(['mean','std']).reset_index()
        plt.figure(figsize=(8,4))
        plt.errorbar(agg['strike_diff_steps'], agg['mean'], yerr=agg['std'],
                    marker='o', linestyle='-')
        plt.title('Mean ± Std of skew_per_step by strike_diff_steps')
        plt.xlabel('strike_diff_steps')
        plt.ylabel('skew_per_step')
        plt.tight_layout()
        plt.show()
    def plot_quantile(self,df):
        # 4. 档位分位数：多重 quantile 曲线
        quantile_levels = [0.05, 0.10, 0.50, 0.90, 0.95]
        q = (
            df
            .groupby('strike_diff_steps')['skew_per_step']
            .quantile(quantile_levels)
            .unstack(level=-1)
            .sort_index()
        )
        plt.figure(figsize=(10,6))
        for qlvl in quantile_levels:
            plt.plot(q.index, q[qlvl], marker='o', label=f'q={qlvl:.2f}')
        plt.title('Quantiles of skew_per_step by strike_diff_steps')
        plt.xlabel('strike_diff_steps')
        plt.ylabel('skew_per_step')
        plt.legend()
        plt.tight_layout()
        plt.show()
    def plot_scatter(self,df):
        # 5. 可选：散点图（带少量抖动）展示原始点
        plt.figure(figsize=(8,4))
        # 为了防止完全重叠，加一点水平抖动
        jitter = (np.random.rand(len(df)) - 0.5) * 0.2
        plt.scatter(df['strike_diff_steps'] + jitter,
                    df['skew_per_step'],
                    alpha=0.3, s=10)
        plt.title('Scatter of skew_per_step vs. strike_diff_steps')
        plt.xlabel('strike_diff_steps')
        plt.ylabel('skew_per_step')
        plt.tight_layout()
        plt.show()


class Signals:
    def __init__(self,quantile) -> None:
        self.quantile=quantile
        
    def get_signals(self,df_skew,window     = 1000, min_periods = 500):
                # 假设 df_skew 已包含三列：
        #   'minute' (pd.Timestamp), 'strike_diff_steps' (int), 'skew_per_step' (float)
        df = df_skew.sort_values(['strike_diff_steps','minute']).reset_index(drop=True)

        # 1. 参数设置
          # 窗口长度（分钟）
          # 最少样本数

        # 2. 计算滚动分位数 & 中位数
        df['q10']   = df.groupby('strike_diff_steps')['skew_per_step'] \
                        .transform(lambda x: x.rolling(window, min_periods).quantile(self.quantile[0]))
        df['q50']   = df.groupby('strike_diff_steps')['skew_per_step'] \
                        .transform(lambda x: x.rolling(window, min_periods).quantile(0.50))
        df['q90']   = df.groupby('strike_diff_steps')['skew_per_step'] \
                        .transform(lambda x: x.rolling(window, min_periods).quantile(self.quantile[-1]))

        # 3. 生成入场信号
        #    Put Spread 信号：skew 从 ≤q90 跃升至 >q90
        #    Call Spread 信号：skew 从 ≥q10 跃降至 <q10
        df['prev_skew'] = df['skew_per_step'].shift(1)
        df['prev_q90']  = df['q90'].shift(1)
        df['prev_q10']  = df['q10'].shift(1)

        df['signal_put']  = (df['prev_skew'] <= df['prev_q90']) & (df['skew_per_step'] > df['q90'])
        df['signal_call'] = (df['prev_skew'] >= df['prev_q10']) & (df['skew_per_step'] < df['q10'])

        # 4. 扫描回归时间
        events = []
        for d, grp in df.groupby('strike_diff_steps'):
            grp = grp.reset_index(drop=True)
            # Put events
            idxs = np.where(grp['signal_put'])[0]
            for i in idxs:
                t0 = grp.loc[i, 'minute']
                sub = grp.loc[i+1:]
                hit = sub[sub['skew_per_step'] <= sub['q50']]
                if not hit.empty:
                    t1 = hit.iloc[0]['minute']
                    events.append({'type':'put', 'steps':d, 't0':t0, 't1':t1, 'dt':(t1-t0).total_seconds()/60})
            # Call events
            idxs = np.where(grp['signal_call'])[0]
            for i in idxs:
                t0 = grp.loc[i, 'minute']
                sub = grp.loc[i+1:]
                hit = sub[sub['skew_per_step'] >= sub['q50']]
                if not hit.empty:
                    t1 = hit.iloc[0]['minute']
                    events.append({'type':'call','steps':d, 't0':t0, 't1':t1, 'dt':(t1-t0).total_seconds()/60})

        events_df = pd.DataFrame(events)
        return events_df
    def plot(self,events_df):
        # 5. 可视化 & 统计

        # 回归时间分布
        plt.figure(figsize=(8,4))
        plt.hist(events_df['dt'], bins=50)
        plt.title('Distribution of Reversion Times (minutes)')
        plt.xlabel('Minutes to Revert')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

        # 平均回归时间按档位
        agg = events_df.groupby(['type','steps'])['dt'].mean().unstack('type')
        plt.figure(figsize=(10,4))
        agg.plot(marker='o')
        plt.title('Mean Reversion Time by strike_diff_steps')
        plt.xlabel('strike_diff_steps')
        plt.ylabel('Mean Reversion Time (min)')
        plt.tight_layout()
        plt.show()

        # 整体平均
        print("Overall mean reversion times (min):")
        print(events_df.groupby('type')['dt'].mean())
        return events_df,agg
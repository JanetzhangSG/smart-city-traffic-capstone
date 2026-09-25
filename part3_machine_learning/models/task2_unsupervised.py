from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SEVERE_WEATHER = {'Fog','Snow','Squall','Thunderstorm'}
LOW_VISIBILITY = {'Fog','Mist','Haze','Smoke'}

def setup_logging(path: Path, level: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    fh = logging.FileHandler(path, mode='a', encoding='utf-8'); fh.setFormatter(fmt)
    sh = logging.StreamHandler(); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

def load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.ParserError, OSError) as exc:
        logger.error('Unable to load input data: %s', exc, exc_info=True)
        raise
    logger.info('Loaded Task 2 input: %d rows x %d columns', *df.shape)
    required = {'date_time','traffic_volume','weather_main'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
    bad = int(df['date_time'].isna().sum())
    if bad:
        logger.warning('Dropping %d rows with invalid date_time values', bad)
        df = df.dropna(subset=['date_time']).copy()
    return df

def add_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['hour'] = out['date_time'].dt.hour
    out['day_of_week'] = out['date_time'].dt.dayofweek
    out['day_type'] = np.where(out['day_of_week'] >= 5, 'Weekend', 'Weekday')
    out['hour_sin'] = np.sin(2*np.pi*out['hour']/24)
    out['hour_cos'] = np.cos(2*np.pi*out['hour']/24)
    out['weather_severity'] = 0
    out.loc[out['weather_main'].isin(LOW_VISIBILITY), 'weather_severity'] = 1
    out.loc[out['weather_main'].isin(SEVERE_WEATHER), 'weather_severity'] = 2
    q1,q2,q3 = out['traffic_volume'].quantile([.25,.5,.75]).values
    logger.info('Congestion quartiles: Q1=%.3f, Q2=%.3f, Q3=%.3f', q1,q2,q3)
    out['congestion_category'] = pd.cut(out['traffic_volume'], [-np.inf,q1,q2,q3,np.inf], labels=['Low','Medium','High','Severe'], include_lowest=True)
    out['time_period'] = pd.cut(out['hour'], [-1,5,9,15,19,23], labels=['Overnight','Morning','Midday','Evening Peak','Late Evening'])
    return out

def run_kmeans(df: pd.DataFrame, outdir: Path, seed: int=42):
    cols = ['hour_sin','hour_cos','weather_severity','traffic_volume']
    X = df[cols].astype(float)
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    sample_n = min(10000, len(df)); rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), sample_n, replace=False)
    scores = {}
    for k in range(2,7):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(Xs)
        score = silhouette_score(Xs[idx], labels[idx])
        scores[k] = score
        logger.info('K-means candidate k=%d silhouette=%.4f', k, score)
    best_k = max(scores, key=scores.get)
    model = KMeans(n_clusters=best_k, random_state=seed, n_init=20)
    labels = model.fit_predict(Xs)
    result = df.copy(); result['cluster'] = labels
    summary = result.groupby('cluster').agg(records=('cluster','size'), avg_hour=('hour','mean'), avg_weather_severity=('weather_severity','mean'), avg_traffic=('traffic_volume','mean')).reset_index()
    summary['share_pct'] = 100*summary['records']/len(result)
    # dominant values for interpretation
    dom_time = result.groupby('cluster')['time_period'].agg(lambda s: s.value_counts().index[0]).rename('dominant_time_period')
    dom_weather = result.groupby('cluster')['weather_main'].agg(lambda s: s.value_counts().index[0]).rename('dominant_weather')
    dom_cong = result.groupby('cluster')['congestion_category'].agg(lambda s: s.value_counts().index[0]).rename('dominant_congestion')
    summary = summary.merge(dom_time, on='cluster').merge(dom_weather,on='cluster').merge(dom_cong,on='cluster')
    summary['interpretation'] = summary.apply(lambda r: f"{r.dominant_time_period} conditions with {r.dominant_congestion} traffic; dominant weather {r.dominant_weather}; average traffic {r.avg_traffic:.0f} vehicles/hour.", axis=1)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir/'task2_cluster_summary.csv', index=False)
    result[['date_time','traffic_volume','weather_main','hour','weather_severity','cluster']].to_csv(outdir/'task2_cluster_assignments.csv', index=False)
    pd.DataFrame({'k':list(scores), 'silhouette':[scores[k] for k in scores]}).to_csv(outdir/'task2_k_selection.csv', index=False)
    logger.info('Selected k=%d with silhouette=%.4f; saved cluster summaries', best_k, scores[best_k])
    return result, summary, scores, best_k

def mine_rules(df: pd.DataFrame, min_support=0.01, min_conf=0.20) -> pd.DataFrame:
    # Targeted association rules: combinations of time period, day type, weather -> congestion category.
    base = df[['time_period','day_type','weather_main','congestion_category']].dropna().astype(str)
    n = len(base)
    targets = sorted(base['congestion_category'].unique())
    target_support = {t:(base['congestion_category']==t).mean() for t in targets}
    antecedent_cols = ['time_period','day_type','weather_main']
    rules=[]
    for r in (1,2,3):
        for cols in combinations(antecedent_cols,r):
            grouped = base.groupby(list(cols), observed=True)
            for keys, g in grouped:
                if not isinstance(keys, tuple): keys=(keys,)
                ant_count=len(g); support_ant=ant_count/n
                if support_ant < min_support: continue
                antecedent=' AND '.join(f'{c}={v}' for c,v in zip(cols,keys))
                counts=g['congestion_category'].value_counts()
                for target,count in counts.items():
                    support=count/n; confidence=count/ant_count
                    if support < min_support or confidence < min_conf: continue
                    lift=confidence/target_support[target] if target_support[target] else np.nan
                    rules.append({'antecedent':antecedent,'consequent':f'congestion_category={target}','support':support,'confidence':confidence,'lift':lift,'antecedent_count':ant_count,'rule_count':int(count)})
    rules_df=pd.DataFrame(rules)
    if rules_df.empty:
        logger.warning('No association rules met thresholds support>=%.3f confidence>=%.3f',min_support,min_conf)
        return rules_df
    rules_df=rules_df.sort_values(['lift','confidence','support'],ascending=False).reset_index(drop=True)
    logger.info('Mined %d congestion association rules; highest lift %.3f',len(rules_df),rules_df.iloc[0]['lift'])
    return rules_df

def save_figures(summary, scores, rules, figdir: Path):
    figdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,4)); plt.plot(list(scores), list(scores.values()), marker='o'); plt.xlabel('Number of clusters (k)'); plt.ylabel('Silhouette score'); plt.title('K-means Cluster Selection'); plt.tight_layout(); p=figdir/'task2_k_selection.png'; plt.savefig(p,dpi=160); plt.close(); logger.info('Figure saved: %s',p)
    plt.figure(figsize=(7,4)); plt.bar(summary['cluster'].astype(str),summary['avg_traffic']); plt.xlabel('Cluster'); plt.ylabel('Average traffic volume'); plt.title('Average Traffic by K-means Cluster'); plt.tight_layout(); p=figdir/'task2_cluster_traffic.png'; plt.savefig(p,dpi=160); plt.close(); logger.info('Figure saved: %s',p)
    if not rules.empty:
        top=rules.head(10).iloc[::-1]
        plt.figure(figsize=(9,5)); plt.barh(range(len(top)),top['lift']); plt.yticks(range(len(top)),[f"{a} -> {c.replace('congestion_category=','')}" for a,c in zip(top['antecedent'],top['consequent'])],fontsize=7); plt.xlabel('Lift'); plt.title('Top Congestion Association Rules by Lift'); plt.tight_layout(); p=figdir/'task2_top_rules.png'; plt.savefig(p,dpi=160); plt.close(); logger.info('Figure saved: %s',p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True,type=Path)
    ap.add_argument('--output-dir',default=Path('.'),type=Path)
    ap.add_argument('--log',default=Path('logs/task2.log'),type=Path)
    ap.add_argument('--log-level',default='INFO',choices=['DEBUG','INFO','WARNING','ERROR'])
    args=ap.parse_args(); setup_logging(args.log,args.log_level)
    logger.info('Part 3 Task 2 unsupervised learning started')
    try:
        df=add_analysis_features(load_data(args.input))
        results=args.output_dir/'results'; figs=args.output_dir/'figures'; models=args.output_dir/'models'
        clustered,summary,scores,best_k=run_kmeans(df,results)
        rules=mine_rules(df)
        rules.to_csv(results/'task2_association_rules.csv',index=False)
        save_figures(summary,scores,rules,figs)
        # Save KMeans metadata rather than pickled scaler/model; Task 2 deliverable focuses on analysis.
        top_txt=args.output_dir/'results'/'task2_interpretation.txt'
        with top_txt.open('w',encoding='utf-8') as f:
            f.write(f'K-means selected k={best_k}.\n\nCluster interpretations:\n')
            for _,r in summary.iterrows(): f.write(f"Cluster {int(r.cluster)}: {r.interpretation}\n")
            f.write('\nHighest-lift congestion rules:\n')
            for _,r in rules.head(10).iterrows():
                f.write(f"{r.antecedent} -> {r.consequent}; support={r.support:.3f}, confidence={r.confidence:.3f}, lift={r.lift:.3f}.\n")
        logger.info('Task 2 completed successfully; outputs saved under %s',args.output_dir)
    except Exception as exc:
        logger.error('Task 2 failed: %s',exc,exc_info=True); sys.exit(1)

if __name__=='__main__': main()

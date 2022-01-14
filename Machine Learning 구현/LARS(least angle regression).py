import pandas as pd
import numpy as np

class LARS():
    def __init__(self):
        return

    def fit(self, x, y):
        x = pd.DataFrame(x)
        col_length = x.shape[1]
        scale_y = y - np.mean(y)
        beta = [0] * col_length
        adj_r = scale_y - np.dot(x, beta) # 현재 시점에서의 잔차
        correlation = abs(pd.DataFrame(np.corrcoef(x.T,adj_r)).iloc[:-1,-1]) # X와 잔차의 corr
        rank_idx = correlation.rank(ascending=False).astype(int) # y와 가장 correlate되어있는 열의 인덱스
        max_cor_idx_1 = rank_idx[rank_idx==1].index
        max_cor_list = [max_cor_idx_1[0]] # corr가 높은 순으로 인덱스를 추가
        alpha = 0.001 # delta 방향으로 (한 스텝당)전진할 길이

        for i in range(col_length):
            new_X = x.iloc[:,max_cor_list]
            delta = np.dot(np.dot(np.linalg.inv(np.dot(new_X.T,new_X)),new_X.T), adj_r) # 계수 중 0이 아닌 변수들의 전진할 방향과 거리
            add_beta = pd.Series([0]*col_length) # 각 계수별로 업데이트(전진)할 거리
            add_beta[max_cor_list] = delta

            if i < col_length-1:
                rank_idx = correlation.rank(ascending=False).astype(int)
                max_cor_idx_next = rank_idx[rank_idx==i+2].index # 다음으로 corr 높은 인덱스 찾기 (업데이트)

                while correlation[max_cor_idx_1].values > correlation[max_cor_idx_next].values : # 계수 max와 그다음 max가 같아질 때까지 업데이트
                    for j in range(len(max_cor_list)): # 각 계수를 작은 스텝 업데이트
                        beta[max_cor_list[j]] = beta[max_cor_list[j]] + alpha * add_beta[max_cor_list[j]]
                        adj_r = scale_y - np.dot(x,beta) # 잔차 수정
                        correlation = abs(pd.DataFrame(np.corrcoef(x.T,adj_r)).iloc[:-1,-1]) # X와 잔차의 상관계수 업데이트

                        if correlation[max_cor_idx_1].values > correlation[max_cor_idx_next].values : # beta를 업데이트하다가 corr의 순서가 바뀌면 인덱스 수정
                            rank_idx = correlation.rank(ascending=False).astype(int)
                            max_cor_idx_next = rank_idx[rank_idx==i+2].index
                max_cor_list.append(max_cor_idx_next[0])
        self.beta_hat = pd.concat([pd.Series(np.mean(y)), beta+add_beta])
        return self.beta_hat

    def predict(self, x):
        x = pd.DataFrame(x)
        x0 = pd.Series([1] * len(x))
        x = pd.concat([x0,x], axis=1)
        return np.dot(x,self.beta_hat)

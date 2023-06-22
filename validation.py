import numpy as np
import pandas as pd
from collections import Counter
from collections import namedtuple, defaultdict
from typing import NamedTuple
from heapq import heappush, heappop

from variable import Variable, Tensor,Generator

import json
class NumpyEncoder(json.JSONEncoder):
    """ 딕셔너리를 json으로 저장할 때 numpy 형태로 인코딩 
    """
    # json serialize numpy
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class Converter():
    """ 딕셔너리를 json으로 저장
    """
    def __init__(self):
        pass
    def to_json(self, obj:dict):
        return json.dumps(obj, cls=NumpyEncoder, indent=4)
    
    def save(self, obj:dict, path:str):
        with open(path, 'w') as f:
            json.dump(obj, f, cls=NumpyEncoder, indent=4)


class DataInfer:
    """## DataInfer
    data의 기술통계량을 추론

    ### args
        `data` (np.ndarray)
            raw 데이터, 현재 pd.DataFrame과 np.ndarray만 지원
        `name` (str), default = None
            데이터 변수명 (dataframe의 컬럼명), None인 경우 0부터 sequence로 지정
        `bins` (int), default = 10
            수치형 변수의 분포 추정 때 나눌 구간 수
        `smoothing` (bool), defaul = False
            수치형 변수의 분포 추정을 연속형으로 할 지 여부
        `unit_of_time` (bool), defaul = 's'
            datetime 측정 단위
    """
    def __init__(self, data, columns=None, bins=10, smoothing=False, unit_of_time='s'):
        if isinstance(data, pd.DataFrame):
            self.var = Variable(data.to_numpy(), name=data.columns.to_list())
        elif isinstance(data, np.ndarray):
            self.var = Variable(data, name=columns)
        else:
            raise TypeError('{} is not supported'.format(type(data)))
        
        self.bins = bins
        self.smoothing = smoothing
        self.unit_of_time = unit_of_time

    def to_tensor(self, dtype):
        names = self.var.dindex(dtype=dtype)[1].tolist()
        if len(names) == 0:
            return None
        return Tensor(self.var.dslice(dtype), name = names)

    def infer_schema(self)->NamedTuple:
        """### infer_schema
        data의 스키마, 데이터 기본 정보를 추론

        ### return
        NamedTuple('schema')
            name : 변수명
            nrow : row 개수
            null count : null 개수
            data type : 데이터 타입
        """
        _fields = ['field', 'n_row', 'n_null', 'dtype']
        res = namedtuple('schema', _fields, defaults=([],)*len(_fields))
                
        return res(
            self.var.name,
            self.var.nrow, 
            self.var.isnull().sum(axis=0),
            self.var.dtype
        )

    def infer_num(self)->NamedTuple:
        """### infer_num
        수치형 (int, float)변수의 기술통계량, bucket을 추론

        ### return
        NamedTuple('num_stat')
            name : 변수명
            mean : 평균
            std : 표준편차
            min : 최소값
            q1 : 1분위수 (q=0.25)
            median : 중간값 (q=0.5)
            q3 : 3분위수 (q=0.75)
            max : 최대값
            mode : 최대빈도값
            skew : 왜도
            kurtosis : 첨도
            bucket : 분포
        """

        _fields = ['field', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'max', 'mode', 'skew', 'kurtosis', 'bucket']
        res = namedtuple('num_stat', _fields, defaults=([],)*len(_fields))

        tensor = self.to_tensor('nums')

        return res(
            tensor.name,
            tensor.mean(axis=0),
            tensor.std(axis=0),
            tensor.min(axis=0),
            tensor.quantile(q=0.25, axis=0),
            tensor.quantile(q=0.5, axis=0),
            tensor.quantile(q=0.75, axis=0),
            tensor.max(axis=0),
            tensor.mode(axis=0),
            tensor.skew(axis=0),
            tensor.kurtosis(axis=0),
            tensor.bucket(self.bins, self.smoothing)
        ) if tensor != None else res()

    def infer_cat(self)->NamedTuple:
        """### infer_cat
        범주형(str) 변수의 기술통계량을 추론

        ### return
        NamedTuple('cat_stat')
            name : 변수명
            lbs : 라벨
            cnt : 각 라벨의 개수
        """
        
        _fields = ['field', 'lbs', 'cnt']
        res = namedtuple('cat_stat', _fields, defaults=([],)*len(_fields))

        tensor = self.to_tensor('str')

        if tensor != None:      
            lbs, cnt = tensor.unique()
            return res(
                tensor.name,
                lbs,
                cnt
            )
        else:
            return res()
        
    def infer_dt(self)->NamedTuple:
        """### infer_dt
        datetime(dt) 변수의 기술통계량을 추론

        ### return
        NamedTuple('cat_stat')
            min : 가장 이른 시간
            max : 가장 늦은 시간
            freq : 주기성
            cycle : 순환성
        """
        
        _fields = ['field', 'start', 'end', 'freq', 'cycle']
        res = namedtuple('datetime', _fields, defaults=([],)*len(_fields))
        
        tensor = self.to_tensor('dt')

        return res(
            tensor.name,
            tensor.astype('O').min(axis=0),
            tensor.astype('O').max(axis=0),
            tensor.freq(self.unit_of_time),
            tensor.cycle(self.unit_of_time)
        ) if tensor != None else res()

    def build(self):
        """### build
        DataInfer 입력 데이터의 추론 결과를 빌드, data의 통계 정보
            - schema : 스키마
            - num_stat : 수치형 변수 통계량
            - cat_stst : 범주형 변수 통계량
            - datetime : 시간 통계량
            
        ### return
            dict (데이터 레포트)
        """
        funcs = {
            'schema': self.infer_schema,
            'num_stat': self.infer_num,
            'cat_stat': self.infer_cat,
            'datetime': self.infer_dt,
        }

        report = {k: f()._asdict() for k, f in funcs.items()}
        
        # ndarray -> list 변환
        for r in report.keys():
            for k , v in report[r].items():
                if isinstance(v, np.ndarray):
                    report[r][k] = v.tolist()

        return report


class DataValidate():
    """## DataValidate
    데이터 유효성을 검증. base 데이터를 기준으로 eval 데이터의 유효성 검사.

    ### args
        `jsd_threshold `: jensen_shannon_divergence의 이상 탐지 임계치
        `lin_threshold `: L-infinite norm의 이상 탐지 임계치
        `num_stat_treshold` : base와 eval의 왜도, 첨도 차이 임계치

            threshold 값을 넘어가면 anomalies로 판단
    ### anomal discripts
        - schema
            - field : base와 eval 데이터의 변수 차이
            - dtype : base와 eval 데이터의 데이터 타입 차이
            - n_null : base와 eval 데이터의 null 값 비율의 차이
                ratio = n_null/n_row

        - num_stat
            - distance : 임계치를 넘어가는 jensen-shannon distance 값
                jensen-shannon distance : 1/2*kld(p||m) + 1/2*kld(q||m), where m=(p+q)/2
            - skew : base와 eval 분포의 기울어진 정도 차이
            - kurtosis : base와 eval 분포의 뾰족한 정도 차이
        
        - cat_stat
            - distance : 임계치를 넘어가는 L-inf norm 값
                L-infinite norm : max(x)
            - lbs : base와 eval의 unique label 값의 차이, eval에 새로운 label이 들어온 경우
            - cnt: base와 eval unique label의 비율 차이
                ratio = base_cnt/eval_cnt 
    """

    def __init__(self):
        self.jsd_threshold = 0.01
        self.lin_threshold = 0.01
        self.num_stat_treshold = 0.5
    
    @property
    def descripts(self):
        return {
            "field": "anoamly fields between base data and eval data.",
            "dtype": "anoamly data types of fields between base data and eval data.",
            "n_null": "absolute difference of null ratio (n_null/n_row) between base data and eval data.",
            "num_stat": {
                "distance": "anomaly jensen-shannon distance over threshold 0.01.",
                "skew": "anomaly skewness over threshold 0.5.",
                "kurtosis": "anomaly kurtosis over threshold 0.5."
            },
            "cat_stat": {
                "distance": "anomaly jensen-shannon distance over threshold 0.01.",
                "unique": "anomaly unique labels between base data and eval data.",
                "count": "anomaly count ratio (base connts/eval counts) each label."
            },
            "threshold": {
                "jsd_threshold": "threshold of jensen-shannon divergence. If exceeded, determined as anomalies.",
                "lin_threshold": "threshold of L-infinite norm. If exceeded, determined as anomalies.",
                "num_stat_treshold": "threshold of numerical stat (skewness, kurotisis) difference between base and eval. If exceeded, determined as anomalies."
            }
        }

    def normalized_count(self, stat, field):
        """
        categorical variable의 normalize count를 계산
        """
        cat_stat = stat['cat_stat']
        cat_idx = cat_stat['field'].index(field) # cat_stat index

        lbs = cat_stat['lbs'][cat_idx]
        cnt = cat_stat['cnt'][cat_idx]

        schema = stat['schema']
        scm_idx = schema['field'].index(field) # schema index

        n_nonmissing = schema['n_row'][scm_idx] - schema['n_null'][scm_idx]
        normed_cnt = {lbs[i]: cnt[i]/n_nonmissing for i in range(len(cnt))}

        return normed_cnt
    
    def l_inf_norm(self, base, eval, field):
        """
        ### l_inf_norm
        categorical 데이터의 L-infinite norm 계산
            L-infinite norm = max(ind normalize_cnt)

        ### parameter
            `base` (dict, DataInfer)
                검증의 기준이 되는 base statistics dict 
            `eval` (dict, DataInfer)
                검증의 대상이 되는 eval statistics dict
            `field` (dict, DataInfer)
                검증할 field (variable)
        
        ### result
            float
        """
        base_cnt = self.normalized_count(base, field)
        eval_cnt = self.normalized_count(eval, field)

        union_label = set(base_cnt.keys()) | set(eval_cnt.keys()) # base와 eval의 label이 다를수도 있으므로 합집합
        
        # l_dist = [abs(base_cnt.get(l, 0) - eval_cnt.get(l, 0)) for l in union_label]
        # return max(l_dist)

        l_dist = []
        for l in union_label:
            heappush(l_dist, -abs(base_cnt.get(l, 0) - eval_cnt.get(l, 0)))
            
        return -1*heappop(l_dist)

    def kld(self, p,q):
        """kullback liebler divergence 계산.
        """
        # idx = np.where((p != 0.0)&(q != 0.0))[0] # 0인 경우 제외
        # p = p[idx]
        # q = q[idx]
        return  np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def jsd(self, p, q):
        """jensen shannon divergence 계산, 분포 차이를 거리 개념으로 측정
        """
        m = 0.5 * (p + q)
        return 0.5 * self.kld(p, m) + 0.5 * self.kld(q, m)
    
    def get_boundary(self, base_bucket, eval_bucket, num_idx):
        """base와 eval의 bucket range들의 합집합
        """
        
        bound_set = set()

        for b, e in zip(base_bucket[num_idx], eval_bucket[num_idx]):
            bound_set.update([b[0], b[1], e[0], e[1]])

        return sorted(list(bound_set))
    
    def add_bucket(self, boundaries, sample_count, range_covered, rebucket):
        """새로운 분포 range에 대해 bucket 추가
        """
        for i in range(len(boundaries)-1):
            new_bucket = {
                'low_value' : boundaries[i], # bucket(히스토그램 bin에 해당하는 데이터 집합)의 low
                'high_value': boundaries[i+1], # bucket의 high
                'sample_count': ((boundaries[i+1] - boundaries[i])/range_covered)*sample_count # bucket의 데이터 개수
            }

            rebucket.append(new_bucket)

    def rebucket(self, bucket, num_idx, boundaries):
        """base와 eval의 union boundary에 대해서 rebucket
        """
        rebuckets = []
        init_idx = 0
        max_idx = len(boundaries) - 1
        bucket_element = (i for i in bucket[num_idx])

        # boundaries를 기준으로 두 개의 분포의 해당하는 bucket 생성 -> 이 후 rebucket
        for b in bucket_element:
            low_value, high_value, sample_count = b
            
            # 작은 경우
            while low_value > boundaries[init_idx]:
                new_bucket = {}
                new_bucket['low_value'] = boundaries[init_idx]
                init_idx += 1
                new_bucket['high_value'] = boundaries[init_idx]
                new_bucket['sample_count'] = 0

                rebuckets.append(new_bucket)

            if low_value == high_value and low_value == boundaries[init_idx]:
                new_bucket = {}
                new_bucket['low_value'] = boundaries[init_idx]
                init_idx += 1
                new_bucket['high_value'] = boundaries[init_idx]
                new_bucket['sample_count'] = sample_count

                rebuckets.append(new_bucket)
                continue
            
            # bin의 boundary 계산
            covered_boundaries = []
            while high_value > boundaries[init_idx]:
                covered_boundaries.append(boundaries[init_idx])
                init_idx += 1
            covered_boundaries.append(boundaries[init_idx])

            if len(covered_boundaries) > 0:
                self.add_bucket(covered_boundaries, sample_count, high_value - low_value, rebuckets)

        for _ in range(init_idx, max_idx):
            new_bucket = {}
            new_bucket['low_value'] = boundaries[init_idx]
            new_bucket['high_value'] = boundaries[init_idx + 1]
            new_bucket['sample_count'] = 0

            rebuckets.append(new_bucket)
        
        return rebuckets

    def get_null_cnt(self, stat, field):
        """data field에 해당하는 데이터의 null 값 개수"""
        field_idx = stat['schema']['field'].index(field)
        return stat['schema']['n_null'][field_idx]

    def jensen_shannon_divergence(self, base, eval, field):
        """
        ### jensen_shannon_divergence
        numerical 데이터의 jensen_shannon_divergence 계산
            jensen_shannon_divergence = 1/2*kld(p||m) + 1/2*kld(q||m), where m=(p+q)/2

        ### parameter
            `base` (dict, DataInfer)
                검증의 기준이 되는 base statistics dict 
            `eval` (dict, DataInfer)
                검증의 대상이 되는 eval statistics dict
            `field` (dict, DataInfer)
                검증할 field (variable)
        
        ### result
            float
        """
        base_bucket = base['num_stat']['bucket']
        eval_bucket = eval['num_stat']['bucket']
        num_idx = eval['num_stat']['field'].index(field)

        boundaries = self.get_boundary(base_bucket, eval_bucket, num_idx)

        base_rebuckets = self.rebucket(base_bucket, num_idx, boundaries)
        eval_rebuckets = self.rebucket(eval_bucket, num_idx, boundaries)

        base_null_cnt = self.get_null_cnt(base, field)
        eval_null_cnt = self.get_null_cnt(eval, field)

        bucket_count = len(boundaries) - 1

        if base_null_cnt > 0 or eval_null_cnt > 0:
            base_rebuckets.append({'low_value': None, 'high_value': None, 'sample_count': base_null_cnt})
            eval_rebuckets.append({'low_value': None, 'high_value': None, 'sample_count': eval_null_cnt})
            bucket_count += 1
        
        base_cnt = np.fromiter((b['sample_count'] for b in base_rebuckets), float)
        eval_cnt = np.fromiter((b['sample_count'] for b in eval_rebuckets), float)
        
        base_bucket_values = base_cnt / base_cnt.sum()
        eval_bucket_values = eval_cnt / eval_cnt.sum()

        return self.jsd(base_bucket_values, eval_bucket_values)

    def distance(self, base, eval):
        """
        ### distance
        전체 field의 distance 계산

        ### parameter
            `base` (dict, DataInfer)
                검증의 기준이 되는 base statistics dict 
            `eval` (dict, DataInfer)
                검증의 대상이 되는 eval statistics dict

        ### result
            dict(keys=['num_sta', 'cat_stat'])
        """
        result = {'num_stat':{}, 'cat_stat':{}}
        
        # report에 null이 있는 경우 set()에서 에러발생 -> 빈 리스트로 변환
        fields = set(eval['schema']['field']) - set(eval['datetime']['field']) 

        for f in fields:
            f_idx = eval['schema']['field'].index(f)
            dtype = eval['schema']['dtype'][f_idx]

            if dtype == 'str':
                result['cat_stat'].update({f:self.l_inf_norm(base, eval, f)})
            elif dtype in ['float', 'int']:
                result['num_stat'].update({f:self.jensen_shannon_divergence(base, eval, f)})
            else:
                continue
        
        return result

    def field_anoaml(self, base, eval):
        """field의 차이 계산        
        """
        res = list(set(eval['schema']['field']) - set(base['schema']['field']))
        if res == []:
            return None
        else:
            return res

    def dtype_anoaml(self, base, eval):
        """data type의 차이 계산        
        """
        ad_set =  list(set(eval['schema']['dtype']) - set(base['schema']['dtype']))

        if ad_set == []:
            return None
        else:
            ad_idx = np.where(np.asarray(eval['schema']['dtype']) == np.asarray(ad_set))[0]
            res = {k:v for k, v in zip(np.asarray(eval['schema']['field'])[ad_idx], np.asarray(eval['schema']['dtype'])[ad_idx])}
            return res

    def null_anoaml(self, base, eval):
        """null ratio의 차이 계산        
        """
        base_nullr = np.asarray(base['schema']['n_null']) / np.asarray(base['schema']['n_row'])
        eval_nullr = np.asarray(eval['schema']['n_null']) / np.asarray(eval['schema']['n_row'])

        abs_nullr = abs(eval_nullr - base_nullr)
        null_idx = np.where(abs_nullr!=0)[0]
        null_list = (eval['schema']['field'][idx] for idx in null_idx)

        res = {k:v for k, v in zip(null_list, abs_nullr[null_idx])}
        if res == {}:
            return None
        else:
            return res
    
    def num_anomal(self, base, eval, distances):
        """numerical data의 차이 계산\n
        distance, skew, kurtosis의 차이
        """
        result = {'distance': None,'skew': None, 'kurtosis': None}

        # jensen-shannon distance 이상값
        num_dist_anomal = {}
        for i in distances['num_stat'].keys():
            if distances['num_stat'][i] > self.jsd_threshold:
                num_dist_anomal.update({i: distances['num_stat'][i]})
            else:
                continue
        
        result['distance'] = num_dist_anomal

        # 왜도, 첨도 이상값
        eval_ns, base_ns = eval['num_stat'], base['num_stat']
        for i in list(result.keys())[1:]:
            if len(eval_ns[i]) != len(base_ns[i]):
                result[i] = 'Cannot calcuate.'
                continue
            abs_diff = abs(np.asarray(eval_ns[i]) - np.asarray(base_ns[i]))
            diff_idx = np.where(abs_diff>self.num_stat_treshold)[0]
            diff_list = (eval_ns['field'][idx] for idx in diff_idx)

            result[i] = {k:v for k, v in zip(diff_list, abs_diff[diff_idx])}

        return result

    def cat_anomal(self, base, eval, distances):
        """numerical data의 차이 계산\n
        distance, lbs, cnt
        """
        result = {'distance':None, 'lbs': None, 'cnt': None}

        # L-infinite norm distance 이상값
        cat_dist_anomal = {}
        for i in distances['cat_stat'].keys():
            if distances['cat_stat'][i] > self.lin_threshold:
                cat_dist_anomal.update({i: distances['cat_stat'][i]})
            else:
                continue
        
        result['distance'] = cat_dist_anomal
        
        # unique한 값의 차이 계산
        anomal_unique = {}

        if base['cat_stat']['field'] != None:
            cat_list = list(set(eval['cat_stat']['field']) - set(eval['datetime']['field']))

            for i in range(len(cat_list)):
                eval_counter = Counter(eval['cat_stat']['lbs'][i])
                base_counter = Counter(base['cat_stat']['lbs'][i])
                diff_cnt = dict(eval_counter - base_counter)

                if diff_cnt != {}:
                    anomal_unique.update({cat_list[i]: diff_cnt})
                else:
                    continue
        
        result['lbs'] = anomal_unique

        anomal_ratio = {}
        for i in cat_dist_anomal.keys():
            linf_idx = eval['cat_stat']['field'].index(i)
            base_uniq_lst = list(base['cat_stat']['lbs'][linf_idx])
            eval_uniq_lst = list(eval['cat_stat']['lbs'][linf_idx])

            intersect_set= set(eval_uniq_lst)&set(base_uniq_lst)

            base_cnt_lst = list(base['cat_stat']['cnt'][linf_idx])
            eval_cnt_lst = list(eval['cat_stat']['cnt'][linf_idx])

            eval_intersect = (eval_cnt_lst[eval_uniq_lst.index(e)] for e in intersect_set)
            base_intersect = (base_cnt_lst[base_uniq_lst.index(e)] for e in intersect_set)

            count_ratio = np.fromiter(eval_intersect, float)/np.fromiter(base_intersect, float)
            anomal_ratio[i] = {k:v for k, v in zip(intersect_set, count_ratio)}

        result['cnt'] = anomal_ratio

        return result

    def dt_anomal(self, base, eval):
        """datetime data의 차이 계산\n
        freq, cycle
        """
        result = {'freq':{}, 'cycle': {}}
        base_dt = base['datetime']
        eval_dt = eval['datetime']    

        from itertools import product
        for i, k in product(range(len(eval_dt['field'])), result.keys()):
            b_seq = np.asarray(base_dt[k][i])
            e_seq = np.asarray(eval_dt[k][i])
            try:
                diff_ar = e_seq - b_seq
            except:
                b_seq = np.where(b_seq==None, 0, b_seq)
                e_seq = np.where(e_seq==None, 0, e_seq)

                diff_ar = e_seq - b_seq
            
            if (diff_ar == np.zeros(b_seq.shape)).all():
                pass
            else:
                result[k].update({eval_dt['field'][i]:list(diff_ar)}) 
                
        return result

    def build(self, base, eval):
        """### build
        데이터 유효성 검증 레포트를 빌드

        ### parameter
            `base` (dict, DataInfer)
                검증의 기준이 되는 base statistics dict 
            `eval` (dict, DataInfer)
                검증의 대상이 되는 eval statistics dict  

        ### return
            dict (데이터 레포트)
        """
               
        anomalies = defaultdict(None)

        anomal_fn = {
                'field': self.field_anoaml,
                'ftype': self.dtype_anoaml,
                'n_null': self.null_anoaml,
                'num_stat': self.num_anomal,
                'cat_stat': self.cat_anomal,
                'datetime': self.dt_anomal
            }
        
        distances = self.distance(base=base, eval=eval)

        for a in anomal_fn.keys():
            if a not in ['num_stat', 'cat_stat']:
                res = anomal_fn[a](base, eval)
            else:
                res = anomal_fn[a](base, eval, distances)
            
            anomalies[a] = res

        return dict(anomalies)
    



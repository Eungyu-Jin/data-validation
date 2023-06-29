import numpy as np
import pandas as pd
import scipy.stats as st
import pywt


from dateutil.parser import parse

class Generator:
    """## Generator
    iterable loop를 저장, generator를 반복적으로 사용 가능

    ### parameter
        `iterator` : iterable 객체 (함수)
            ex) Generator(lambda: (x for x in range(0, 100))) 
    """
    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return self.iterator()

class Variable:
    """
    ## Variable
    data의 variable array를 정의  

    ### args
        data (np.ndarray) : 데이터 array
        name (str) : 데이터 변수명, dataframe의 컬럼명    
    """
    def __init__(self, data:np.ndarray, name=None):       
        matching ={
            'float': float,
            'int': int,
            'str': str,
            'dt': object
        } 
        if name==None:
            name = [str(i) for i in range(np.size(data, 1))]
        self.name = name
        self.dtype = self._dtype(data)
        self.data = data if data.ndim > 1 else data.astype(matching[self.dtype]) # 1d array인 경우 바꾸어줌

    @property
    def nrow(self):
        return len(self.data) if self.data.ndim==1 else [len(self.data) for _ in range(np.size(self.data, 1))]

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def T(self):
        return self.data.T

    def __len__(self):
        return len(self.data)
    
    def _generate_ar(self, dtype=None):
        """
        null을 제외한 generate array 생성  
        ndim이 1인 경우 2로 늘려서 생성
        """
        if dtype == None:
            ar = self.data.copy()
        else: 
            ar = self.dslice(dtype)

        if ar.ndim == 1:
            ar = np.expand_dims(ar, 1)

        for i in range(np.size(ar, 1)):
            var = ar[:,i]
            yield var[pd.notnull(var)]

    def _num_ar(self):
        """
        기술통계 함수에 사용하는 nums 자료형 dslice  
        """
        if 'str' in self.dtype or 'dt' in self.dtype:
            ar = self.dslice('nums')
            return ar
        else:
            ar = self.data.copy()
            return ar
        
    def _dtype(self, data):
        """
        data type 추론  
        """
        output = []
        if data.ndim==1:
            data = np.expand_dims(data, 1)
        
        gen_ar = (data[:,i] for i in range(np.size(data, 1)))
        for a in Generator(lambda: gen_ar):
            if len(a) == 0:
                _dtype = 'null'
            
            check_value = a[0]

            if type(check_value) in (int, np.int64, np.int32):
                _dtype = 'int'
            elif type(check_value) in (float, np.float64, np.float32):
                _dtype = 'float'
            elif type(check_value) in (str, np.str_):
                try:
                    int(check_value)
                except:
                    try:
                        parse(check_value)
                        _dtype = 'dt'
                    except:
                        _dtype = 'str'
            else:
                _dtype = 'dt'

            output.append(_dtype)
        
        return output if len(output)>1 else output[0]

    def dindex(self, dtype, names=True):
        """### dindex
        dtype에 해당하는 index 리스트 찾음, array's ndim > 2d 

        ### args
            `dtype` (str) | 'all', 'int', 'float', 'str', 'dt'

            `name` (bool), default = True
                dtype에 해당하는 변수명 return 여부

        ### return
            index list of dtype
        """
        _dtypes = np.array(self.dtype)

        if self.data.ndim == 1:
            ar = np.expand_dims(self.data, 1)
            # raise Exception('array ndim must greater than 1 using dindex.')

        if dtype == 'all':
            idx = np.arange(0, np.size(ar, 1))
        elif dtype =='nums':
            idx = []
            for i in ['int', 'float']:
                idx.append(np.where(_dtypes==i)[0])
            idx = np.concatenate(idx)
        else:
            idx = np.where(_dtypes==dtype)[0]

        # idx = idx if len(idx)>1 else idx[0]

        if names:            
            return idx, np.array(self.name)[idx]
        else:
            return idx

    def dslice(self, dtype):
        """### dslice
        데이터 타입 맞게 변환 후 해당 array 리턴, ndim > 2d 이상

        ### parameter
            `dtype` : 찾을 dtype | 'int', 'float', 'str', 'dt'
                'dt'는 str로 반환

        ### return
            np.ndarray
        """
        dtype_bags = {
            'int': int,
            'float': float,
            'str': str,
            'dt': str,
            'nums': float
        }
        ar = self.data.copy()
        if ar.ndim == 1:
            ar = np.expand_dims(self.data, 1)
        #     raise Exception('array ndim must greater than 1 using dslice.')
        
        idx = self.dindex(dtype, names=False)
        ar = ar[:, idx].astype(dtype_bags[dtype]).squeeze()

        return ar

    def astype(self, dtype):
        """### astype
        data의 dtype 변환

        ### parameter
            `dtype` (str, type) 
                변환할 dtype, numpy의 astype과 동일

        ### return
            np.ndarray
        """
        return self.data.astype(dtype)

    def size(self, axis = None):
        """### size
        data의 size를 구하는 함수

        ### parameter
            `axis` (int), default = None
                구하려는 size의 축

        ### return
            int
        """
        return np.size(self.data, axis=axis)

    def reshape(self, *shape):
        """### reshape
            data를 reshape 해주는 함수

        ### parameter
            `*shape` (tuple)
                reshape할 형태, reshap 이전 배열의 size가 같아야 함

        ### return
            np.ndarray
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return self.data.reshape(shape)

    def transpose(self, *axes):
        """### transpose
        array의 배열을 transpose 해주는 함수

        ### parameter
            `*axes` (tuple)
                transpose할 형태, transpose 이전 배열의 size가 같아야 함\n
                axes에 따라서 배열의 전치가 가능

        ### return
            np.ndarray
        """
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return self.data.transpose(axes)

    def sum(self, axis=None, keepdims=False):
        """### sum
        array의 axis에 따른 합

        ### parameter
            `axis` (int), default = None
                axis = None : 전체 배열에 대해 계산
                axis = 0 : row(x axis) 기준, variable로 계산할 때
                axis = 1 : column(y axis) 기준
                axis = 2 : depth(z axis) 기준

            `keepdims` (bool), default = False
                array의 기존 배열의 유지할지 여부

        ### return
            axis = None : int, float
            axis = int : np.ndarray
        """
        ar = self._num_ar()
        return ar.sum(axis=axis, keepdims=keepdims)

    def mean(self, axis=None, ignore_na=True, keepdims=False):
        """### mean
        array의 axis에 따른 평균, 해당 array에서의 기대값 (대표하는 값)

        ### parameter
            `axis` (int), default = None
                axis = None : 전체 배열에 대해 계산
                axis = 0 : row(x axis) 기준, variable로 계산할 때
                axis = 1 : column(y axis) 기준
                axis = 2 : depth(z axis) 기준

            `ignore_na` (bool), default = True
                True : na값을 제외하고 계산
                False : na값을 포함하고 계산, na값이 있는 경우 nan 리턴

            `keepdims` (bool), default = False
                array의 기존 배열의 유지할지 여부

        ### return
            axis = None : int, float
            axis = int : np.ndarray
        """
        ar = self._num_ar()        
        if ignore_na:
            return np.nanmean(ar, axis= axis, keepdims=keepdims)
        else:
            return np.mean(ar, axis= axis, keepdims=keepdims)

    def std(self, axis=None, ignore_na=True, keepdims=False):
        """### std
        array의 axis에 따른 표준편차, 
        해당 array가 기대값을 기준으로 얼마나 퍼져있는지 나타내는 값

        ### parameter
            `axis` (int), default = None
                axis = None : 전체 배열에 대해 계산
                axis = 0 : row(x axis) 기준, variable로 계산할 때
                axis = 1 : column(y axis) 기준
                axis = 2 : depth(z axis) 기준

            `ignore_na` (bool), default = True
                True : na값을 제외하고 계산
                False : na값을 포함하고 계산, na값이 있는 경우 nan 리턴

            `keepdims` (bool), default = False
                array의 기존 배열의 유지할지 여부

        ### return
            axis = None : int, float\n
            axis = int : np.ndarray
        """
        ar = self._num_ar()
        if ignore_na:
            return np.nanstd(ar, axis= axis, keepdims=keepdims)
        else:
            return np.std(ar, axis= axis, keepdims=keepdims)

    def max(self, axis=None, ignore_na=True, keepdims=False):
        """### max
        array의 axis에 따른 최대값

        ### parameter
            `axis` (int), default = None
                axis = None : 전체 배열에 대해 계산
                axis = 0 : row(x axis) 기준, variable로 계산할 때
                axis = 1 : column(y axis) 기준
                axis = 2 : depth(z axis) 기준

            `ignore_na` (bool), default = True
                True : na값을 제외하고 계산
                False : na값을 포함하고 계산, na값이 있는 경우 nan 리턴

            `keepdims` (bool), default = False
                array의 기존 배열의 유지할지 여부

        ### return
            axis = None : int, float
            axis = int : np.ndarray
        """
        ar = self._num_ar()
        # ar = self.data.copy()
        
        if ignore_na:
            return np.nanmax(ar, axis= axis, keepdims=keepdims)
        else:
            return np.max(ar, axis= axis, keepdims=keepdims)

    def min(self, axis=None, ignore_na=True, keepdims=False):
        """### min
        array의 axis에 따른 최소값
        
        ### parameter
            `axis` (int), default = None
                axis = None : 전체 배열에 대해 계산
                axis = 0 : row(x axis) 기준, variable로 계산할 때
                axis = 1 : column(y axis) 기준
                axis = 2 : depth(z axis) 기준

            `ignore_na` (bool), default = True
                True : na값을 제외하고 계산
                False : na값을 포함하고 계산, na값이 있는 경우 nan 리턴

            `keepdims` (bool), default = False
                array의 기존 배열의 유지할지 여부

        ### return
            axis = None : int, float
            axis = int : np.ndarray
        """
        ar = self._num_ar()
        # ar = self.data.copy()

        if ignore_na:
            return np.nanmin(ar, axis= axis, keepdims=keepdims)
        else:
            return np.min(ar, axis= axis, keepdims=keepdims)

    def quantile(self, q, axis=None, ignore_na=True, keepdims=False):
        """### quantile
        array의 axis에 따른 분위수값, 0.0~1.0의 분위(percentage)에 해당하는 값을 계산\n
        분위 : sample의 관측치를 동일한 연속 간격으로 나누는 값
            ex) 0~100의 범위의 sample이 sample이 있을 때, 0.25 분위에 해당하는 값은 25
        
        ### parameter
            `q` (float)
                quantile 값 (percentage 값)

            `axis` (int), default = None
                axis = None : 전체 배열에 대해 계산
                axis = 0 : row(x axis) 기준, variable로 계산할 때
                axis = 1 : column(y axis) 기준
                axis = 2 : depth(z axis) 기준

            `ignore_na` (bool), default = True
                True : na값을 제외하고 계산
                False : na값을 포함하고 계산, na값이 있는 경우 nan 리턴

            `keepdims` (bool), default = False
                array의 기존 배열의 유지할지 여부

        ### return
            axis = None : int, float
            axis = int : np.ndarray
        """
        ar = self._num_ar()
        if ignore_na:
            return np.nanquantile(ar, q=q, axis= axis, keepdims=keepdims)
        else:
            return np.quantile(ar, q=q, axis= axis, keepdims=keepdims)

    def skew(self, axis=None, ignore_na=True, keepdims=False):
        """### skew
        array의 axis에 따른 왜도
        왜도 : sample 분포의 비대칭성을 나타내는 값, |skew|<3 이면 정규분포와 비슷
            왜도 > 0 : 왼쪽으로 치우친 분포 
            왜도 < 0 : 오른쪽으로 치우친 분포
        
        ### parameter
            `axis` (int), default = None
                axis = None : 전체 배열에 대해 계산
                axis = 0 : row(x axis) 기준, variable로 계산할 때
                axis = 1 : column(y axis) 기준
                axis = 2 : depth(z axis) 기준

            `ignore_na` (bool), default = True
                True : na값을 제외하고 계산
                False : na값을 포함하고 계산, na값이 있는 경우 nan 리턴

            `keepdims` (bool), default = False
                array의 기존 배열의 유지할지 여부

        ### return
            axis = None : int, float
            axis = int : np.ndarray
        """
        ar = self._num_ar()
        nan_policy = 'omit' if ignore_na else 'propagate'
        return st.skew(ar, axis=axis, nan_policy=nan_policy, keepdims=keepdims)

    def kurtosis(self, axis=None, ignore_na=True, keepdims=False):
        """### kurtosis
        array의 axis에 따른 첨도
        첨도 : sample 분포의 꼬리가 두꺼운 정도를 나타내는 값, |kurtosis|<3 이면 정규분포와 비슷/
            첨도 > 3 : 정규분포보다 꼬리가 두껍다
            첨도 < 3 : 정규분포보다 꼬리가 얇다
        
        ### parameter
            `axis` (int), default = None
                axis = None : 전체 배열에 대해 계산
                axis = 0 : row(x axis) 기준, variable로 계산할 때
                axis = 1 : column(y axis) 기준
                axis = 2 : depth(z axis) 기준

            `ignore_na` (bool), default = True
                True : na값을 제외하고 계산
                False : na값을 포함하고 계산, na값이 있는 경우 nan 리턴

            `keepdims` (bool), default = False
                array의 기존 배열의 유지할지 여부

        ### return
            axis = None : int, float
            axis = int : np.ndarray
        """
        ar = self._num_ar()
        nan_policy = 'omit' if ignore_na else 'propagate' # scipy
        return st.kurtosis(ar, axis=axis, nan_policy=nan_policy, keepdims=keepdims)

    def mode(self, axis=None, keepdims=False):
        """### mode
        array의 axis에 따른 최빈값, 가장 빈도가 많은 값 \n
        nan 값은 내부에서 자동으로 제외
        
        ### parameter
            `axis` (int), default = None
                axis = None : 전체 배열에 대해 계산
                axis = 0 : row(x axis) 기준, variable로 계산할 때
                axis = 1 : column(y axis) 기준
                axis = 2 : depth(z axis) 기준

            `keepdims` (bool), default = False
                array의 기존 배열의 유지할지 여부

        ### return
            axis = None : int, float    
            axis = int : np.ndarray
        """
        return st.mode(self.data, axis=axis, nan_policy='omit', keepdims=keepdims).mode

    def isnull(self):
        """### isnull
        array의 null값을 확인

        ### return
            np.ndarray -> True/False
        """
        return pd.isnull(self.data)

    def histogram(self, bins, density=False):
        """### histogram
        연속형 sample 분포의 히스토그램을 계산, 특정 구간 (bin) 안에 포함되는 sample 개수
        
        ### parameter
            `bins` (int)
                sample을 일정 구간으로 나눌 개수

            `density` (bool), default = False
                sample 분포를 연속 확률 분포로 계산 (확률 밀도)

        ### return
            (count, edge)
        """
        ars = Generator(lambda: self._generate_ar(dtype='nums'))
        output = []
        for a in ars:
            cnt, edge = np.histogram(a, bins=bins, density=density)
            output.append((cnt, edge))
        
        return output

    def kde(self):
        """### kde (kernel density estimator)
        gaussain kernel 함수로 연속형 확률 밀도를 추정
        
        ### parameter
            `bins` (int)
                sample을 일정 구간으로 나눌 개수

            `density` (bool), default = False
                sample 분포를 연속 확률 분포로 계산 (확률 밀도)

        ### return
            (kdes, )
        """
        ars = Generator(lambda: self._generate_ar(dtype='nums'))
        output = []
        for a in ars:
            output.append(st.gaussian_kde(a))
        return output

    def density(self, bins):
        """### density
        연속형 sample 분포를 추정
        
        ### parameter
            `bins` (int)
                sample을 일정 구간으로 나눌 개수

        ### return
            (count, edge)
        """
        ars = Generator(lambda: self._generate_ar(dtype='nums'))
        kdes = self.kde()

        output = []
        for k, a in zip(kdes, ars):
            cover_range = (a.max()-a.min())/bins
            cover_init= a.min()

            edge = []
            for _ in range(bins+1):
                edge.append(cover_init)
                cover_init += cover_range

            len_ar = len(a)
            pdf = k.integrate_box_1d
            cnt = [len_ar*pdf(edge[v], edge[v+1]) for v in range(bins)]

            output.append((cnt, edge))

        return output

    def bucket(self, bins, smoothing=False):
        """### bucket
        data의 bucket을 계산, bin 개수에 따라 나눈 각 sample의 수과 경계값 리턴
        
        ### parameter
            `bins` (int)
                sample을 일정 구간으로 나눌 개수

            `smoothing` (bool), default = False
                True : 커널밀도추정으로 연속헝 확률분포로 계산
                False : 히스토그램으로 이산형 확률분포로 계산

        ### return
            bucket : [(low_value, high_value, sample_count)]
        """
        _hist = self.density(bins) if smoothing else self.histogram(bins)

        output = []
        for h in _hist:
            cnt, edge = h
            output.append([(edge[i], edge[i+1], cnt[i]) for i in range(len(cnt))])

        return output if len(output)>1 else output[0]

    def unique(self, return_counts=True):
        """### unique
        data의 개별 값 (labels)과 그 개수(count)를 계산
        
        ### parameter
            `return_counts` (bool), default = True
                count값 리턴 여부

        ### return
            (labels, count)
        """
        ar = np.where(pd.notnull(self.data), self.data, 'null')
        if ar.ndim == 1:
            ar = np.expand_dims(ar,1)

        _lbs, _cnt = [], []
        for i in range(np.size(ar, 1)):
            if return_counts:
                lbs, cnt = np.unique(ar[:,i], return_counts=True)
                _lbs.append(lbs.squeeze().tolist())
                _cnt.append(cnt.squeeze().tolist())
            else:
                lbs = np.unique(ar[:,i], return_counts=False)
                _lbs.append(lbs.squeeze().tolist())
    
        if len(_lbs) == 1:
            _lbs, _cnt = _lbs[0], _cnt[0]

        return _lbs, _cnt if return_counts else _cnt
    
    def _time_attr(self, ar, unit = 's'):
        """시간 데이터의 속성을 계산하는 함수.

        return (delta_ar, forward_idx, inverse_idx)
            delta_ar : 시간 간격의 차이
            foward_idx : 시간이 진행하고 있는 index
            inverse_idx : 시간이 역진행하고 있는 index (순환 데이터인 경우)
        """
        dt_ar = pd.to_datetime(ar).to_numpy()
        delta_ar = np.diff(dt_ar) / np.timedelta64(1, unit)

        stand_idx = np.isclose(delta_ar, 0.0, rtol=0.0, atol=1e-1)
        nonstand_idx = np.where(stand_idx==False)[0]
        inverse_idx = np.where(delta_ar<0.0)[0]
        forward_idx = np.asarray(list(set(nonstand_idx) - set(inverse_idx)))

        return (delta_ar, forward_idx, inverse_idx)

    def freq(self, unit='s'):
        """### freq
        시간 데이터의 주기성을 추론\n
        주기값으로 추정되는 값들의 평균과 최대빈도값 계산

        ### return
            (mean freq, mode freq)
        """
        ars = Generator(lambda: self._generate_ar(dtype='dt'))
        output=[]
        for a in ars:
            delta_ar, forward_idx, _ = self._time_attr(a, unit=unit)
            forward_ar = delta_ar[forward_idx]
            output.append((np.mean(forward_ar), np.unique(forward_ar).max()))
        
        return output
        
    def cycle(self, unit='s'):
        """### freq
        시간 데이터의 순환성을 추론\n
        순환값으로 추정되는 값들의 평균과 최대빈도값 계산

        ### return
            (mean cycle, mode cycle)
        """
        ars = Generator(lambda: self._generate_ar(dtype='dt'))
        output=[]
        for a in ars:
            _, _, inverse_idx = self._time_attr(a, unit=unit)
            interval_idx = np.diff(inverse_idx)
            if len(interval_idx) > 0:
                output.append((np.mean(interval_idx), np.unique(interval_idx).max()))
            else:
                output.append((None, None))
        return output

    def __repr__(self):
        if self.data is None:
            return f'Variable(None),\n name={self.name}'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'Variable({p}), name={self.name})'


class Tensor(Variable):
    """
    ## Tensor
    data의 variable tensor를 정의\n
    데이터 가공에는 tensor로 사용

    ### args
        data (np.ndarray) : 데이터 array
        name (str) : 데이터 변수명, dataframe의 컬럼명    
    """
    def __repr__(self):
        if self.data is None:
            return f'Tensor(None),\n name={self.name}'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'Tensor({p}),\n name={self.name})'
    
    def fft(self, fs, n = 'auto', top = 10, as_dict= True):
        ar = self._num_ar()
        if n=='auto':
            N = len(ar)
        else:
            N = n
        signal = ar[:N, :]

        fs = 10
        T = 1/fs
        Nhalf = int(N/2)

        y = np.fft.fft(signal)
        _amp = np.abs(y)*(2/N) 
        _freq = np.fft.fftfreq(N, T) 

        amp = _amp[range(Nhalf)] 
        freq = _freq[range(Nhalf)] 
        ampsort = np.argsort(-amp, axis=0)
        main_freq = np.mean([freq[ampsort[i]] for i in range(top)], axis=0)

        if as_dict:
            return {
                'amp': amp,
                'freq': freq,
                'main_freq':main_freq
            }
        else:
            return [amp, freq, main_freq]

    def _cwt(self, idxs='auto', wavelet='ricker', widths=[1.0]):      
        from scipy import signal
        if wavelet =='ricker':
            wavelet = signal.ricker
        if idxs=='auto':
            idxs = self.dindex('float', names=False)
        return np.stack(
            [signal.cwt(self.data[:,idx], wavelet=signal.ricker, widths= widths).squeeze() for idx in idxs]
            , axis=1).squeeze()

    def cwt(self, idxs='auto', wavelet='mexh', scale=[1.0, 1.1, 1.2], fs=10):
        ts= []
        T = 1/fs
        if idxs=='auto':
            idxs = self.dindex('float', names=False)
        for i in idxs:
            cwtcoef, _ = pywt.cwt(self.data[:,i], scales=scale, sampling_period=T, wavelet=wavelet)
            ts.append(cwtcoef)

        ts = np.stack(ts,axis=-1)
        
        return ts

    def dwt(self, idxs='auto', wavelet='haar', scale=0.1):
        ts = []
        if idxs=='auto':
            idxs = self.dindex('float', names=False)
        for i in idxs:
            ca, cd = pywt.dwt(self.data[:,i], wavelet=wavelet) # ca는 approximate로 저주파, cd는 detail로 고주파
            cat = pywt.threshold(ca, scale*np.std(ca), mode="soft") # 가운데 임계치 값을 조정해서 분석
            cdt = pywt.threshold(cd, scale*np.std(cd), mode="soft") # 가운데 임계치 값을 조정해서 분석
            ts.append(pywt.idwt(cat, cdt, wavelet)[:-1])

        ts = np.stack(ts,axis=1)

        return ts
    
    def get_waveltfn(self, fn):
        if fn in ['dwt', 'discrete']:
            return pywt.wavelist(kind='discrete')
        elif fn in ['cwt', 'continuous']:
            return pywt.wavelist(kind='continuous')
        else:
            raise ValueError("Not supported fn.")


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

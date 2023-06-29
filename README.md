# data-validator

custom 데이터 검증 모듈

### variable
데이터셋 변수 저장, Tensor 정의

### validation
1. DataInfer
  - 데이터 추론
  - 데이터 추론 결과 report -> json
2. Datavalidate
  - 데이터 검증
  - 수치형 : jensen-shannon divergence
      -> 데이터셋마다 결과가 다른 issue 있음. 업데이트 필요.
  - 문자형 : L-infinite norm
  - TFDV (tensorflow data-validation) 알고리즘 참고

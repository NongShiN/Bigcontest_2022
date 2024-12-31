# 빅콘테스트 데이터분석분야 퓨처스부문

## - 대회 문제
### **앱 사용성 데이터를 통한 대출신청 예측분석**

- **가명화된 데이터를 기반으로 고객의 대출상품 신청여부 예측**
    
    **(2022년 3~5월 데이터제공 / 2022년 6월 예측)   <한도조회 일시 기준 !!>**
    
- **예측모델을 활용하여 탐색적 데이터 분석 수행**
- **대출신청, 미신청 고객을 분류하여 고객의 특성 분석결과 도출**
- **승인된 상품정보 기준으로 하나 이상을 선택하여 대출을 실행한 고객을 예측하는 것**

![image](https://user-images.githubusercontent.com/100005890/209901787-b918b7eb-950b-4ee1-a415-d1b5a98e01d7.png)
![image](https://user-images.githubusercontent.com/100005890/209901796-e21f68e9-19ce-46c6-9536-71feba119e5e.png)



## - 분석 방법
### 0. 분석 목표

- 머신러닝 & 마케팅 자동화를 통한 반복 업무 수행 시간 단축
- 효율적인 시간 자원 활용 / 전략적 마케팅 관리에 투자
- 모델링을 통해 얻은 SHAP Value 해석을 통해 핀다의 프로모션 방향을 구체화
- 대출서비스를 이용한 고객과 이용하지 않은 고객으로 나누고, 군집화를 통해 군집을 나눈 후 각 군집 별 특성을 도출해내어 군집별 메시지를 제안

### 1. 데이터 탐색

- 데이터 병합
    - 공통된 application_id를 가지고 있는 user_spec과 loan_result의 데이터들을 하나의  dataframe으로 합침
    → target data인 is_applied 칼럼이 loan_result에만 있으므로 loan_result를 기준으로 합침. 이때 데이터는 각각의 application_id가 신청 가능한 bank_id 개수만큼 존재
- train, test로 분리하여 새로운 csv파일 생성
    - 2022년 3,4,5월 데이터를 train data로 분리 (bigcon_train_origin.csv 생성)
    - is_applied가 결측치인 2022년 6월 데이터를 test 데이터로 분리 (bigcon_test_origin.csv 생성)
- ~~시각화~~ → 데이터 분포 확인
    - target 데이터의 분포를 시각화를 통해 확인
    - 범주형 변수들과 is_applied의 관계를 시각화를 통해 확인
    
    > income_type, gender, employment_type, purpose, houseown_type, personal_rehabilitation_yn, personal_rehabiltiation_complete_yn, bank_id, product_id
    > 
    - 수치형 변수들과 is_applied의 관계를 시각화를 통해 확인
    
    > birth_year : age로 만든 후 확인해보기로 결정 
    insert_time : 필요없을 것 같아서 보지 않음
    credit_score, yearly_income, company_enter_month : 
      company_enter_month > cont_month로 만든 후 확인해보기로 결정
    desired_amount, existing_loan_cnt, existing_loan_amt, loanapply_insert_time : 필요없을 것 같아서 보지 않음
    loan_limit, loan_rate :
    > 
- 상관관계
    - 각 변수들 간의 상관관계 확인

### 2. 데이터 전처리

- 파생변수 생성
    1. cont_month(근속월수)
        
        loanapply_insert_time을 이용하여 month(월)과 day(일)로 분리, 
        month칼럼과 company_enter_month칼럼의 차이를 계산해 근속월수 생성
        
    2. ~~cert_count(본인인증 횟수)~~
        
        ~~각 user_id 당 ‘event’칼럼의 행동명 중 ‘CompleteIDCertification’의 횟수를 셈~~
        
    3. admission_count (승인 상품 개수)
        
        각 application_id 당 승인된 상품 개수                              
        특정 application_id의 개수가 곧 product_id의 개수(승인된 상품 개수)이므로 application_id가 중복된 개수를 셈
        
    4. over_desired(희망 금액 대비 대출 한도 초과 여부)
        
        희망 대출 금액과 대출 한도 금액을 비교하여 desired_amount(희망 대출 금액)보다 loan_limit(대출 한도 금액)가 클 경우 1, 그렇지 않을 경우 0의 값을 할당
        
    5. age_level
        
        birth_year를 2022년 기준으로 나이(age)로 변환, 이후에 20세부터 94세까지 5세 단위로 나누어 총 14개로 범주화
        
- 결측치 처리
    - train,test 공통으로 적용
        - personal_rehabilitation_yn : 결측치인 경우 개인회생자가 아니기 때문에 입력하지 않았다고 판단하여 0으로 채움
        - personal_rehabilitation_complete_yn : 회생자 여부가 0이면서 납입완료여부가 1인 경우가 존재할 수 없으므로 0으로 바꿔줌, 나머지 결측치는 납입완료여부를 입력할 필요가 없기 때문에 비어있는 자료라 판단하여 0으로 채워줌
        - existing_loan_cnt : 최소값이 1이기 때문에 결측치는 기대출이 없는 값이라고 판단하여 0으로 채워줌
        - existing_loan_amt : 기대출수가 결측치이면 기대출금액도 결측치이기 때문에 0으로 채움, 나머지 900292개의 결측치는 IterativeImputer를 이용해 채움
        - insert_time, loanapply_insert_time은 불필요하다고 판단하여 제거
        - crediet_score 칼럼은 결측치가 많았고, is_applied과 연관성이 있다고 판단하여 credit score와 상관계수가 0.2 이상인 칼럼들로 IterativeImputer를 이용해 credit_score의 결측치를 채움
    - train만 적용
        - loan_rate, loan_limit 컬럼에 결측치(nan)가 있는 경우는 금융사에서 값을 보내주지 않은 경우로, 해당 경우는 채점에서 제외할 예정이니 제거
        - train의 나머지 특정 값으로 채우기 어렵다고 판단되는 결측치가 있는 행 제거
    - test만 적용
        - test의 나머지 특정 값으로 채우기 어렵다고 판단되는 결측치가 있는 행 IterativeImputer를 이용해 채움
- 인코딩
    - 범주형 변수에 대해 label encoding 진행
    
    > income_type, employment_type, houseown_type, purpose, age_level
    > 
    
- 이상치 처리
    - boxplot을 통해 이상치를 확인한 후 이상치라고 판단되는 데이터 삭제
    - train 중 비정상적인 값으로 판단된 데이터 삭제
    - 입사년월이 22년 6월 이후인 데이터
    - 생년과 입사년도의 차가 20년 미만인 데이터
- 스케일링
    - 범위가 큰 변수들에 대해 log 변환
        
        > credit_score, yearly_income, desired_amount, existing_loan_amt, loan_limit
        > 
        
    

### 3. 모델링

- train_test_split
    - 성능 평가를 위한 X_train, y_train, X_test, y_test 생성
- 머신러닝 모델
    - CatBoost, LightGBM 등 다양한 Tree기반 모델 사용 후 여러가지 앙상블을 시도해 본 뒤 성능이 가장 높은 모델 채택
    - target이 불균형한 분포를 보이므로 파라미터를 통한 불균형 처리를 진행
    - Parameter Tuning
        - 베이지안 최적화를 통해 하이퍼파라미터 튜닝 (Bayesian Optimization)
    - 불균형 데이터이기 때문에 threshold 변경을 통해 조금 더 보수적으로 대출 신청 여부 판단
- 해석
    - SHAP Value를 통해 예측에 중요하게 영향을 미치는 변수 확인
    - 모델 성능 평가

### 4. 군집화

- 추가 전처리
    - 데이터 줄이기
        - 이전에 전처리한 데이터는 하나의 application_id에 대해 admission_count 개수만큼 데이터가 존재. 이때 admission_count가 많은 사람의 정보가 군집화 할 때 많이 반영되는 것을 막기 위하여 신청서 당 사용자 개인정보 & 신청서 입력 정보가 중복되는 데이터들을 삭제
            - 신청 이력이 있는(is_applied가 1인 데이터가 존재하는) 신청서의 경우 is_applied가 1인 행만 남김
            - is_applied가 0만 있는 신청서의 경우 결국 신청하지 않은 사람이므로 한 행만 남기고 나머지 제거
    - service_dummy 생성 후 이상치 처리
        - 각 user_id 당 log_data의 event 칼럼의 각 행동명들의 행동횟수를 센 뒤 
        'UseLoanManage', 'UsePrepayCalc', 'UseDSRCalc', 'GetCreditInfo' 값만 더함
- 대출 신청한 사람들과 대출 신청하지 않은 사람들 분리
- PCA
    - column 선택 후 스케일링
    - 시각화를 하기 위해 2차원으로 축소
- KMeans
    - Elbow point 확인 후 알맞은 군집 수를 정해 클러스터링
- 해석


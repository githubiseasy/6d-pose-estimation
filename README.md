sh # 객체 3D 데이터 유효성 검증
## https://github.com/DongGeun-Yoon/nia-ssp
<br>

## * 전체 실행 (build & test)
> git clone https://github.com/DongGeun-Yoon/nia-ssp <br>
> ./build.sh <br>
> ./run.sh all <br>
<br>

## * 객체별 샘플링 평가
### 1. 폴더 구성
````
├──── 시험 환경(experimental_enviroment)
├──── 시험 결과(experimental_results)
├──── 도커이미지(docker_image) <----- working directory
│    ├──── data (500개 모델 weight)
│    ├──── cfg
│    ├──── doc
├──── 평가용 데이터셋(test_datasets)
````

### 2. docker image 생성 --> build.sh
> ./build.sh <br>

(참고) docker image 로드: docker load -i nia-ssp.tar <br>
(참고) docker image 저장: docker save -o ./nia-ssp.tar nia-ssp:0.1 <br>
<br>


### 3. 객체 ID 별 유효성 검증 --> test.sh
> ./test.sh 010118 <br>

(평가 결과/로그 파일) ./experimental_results/ <br>
<br>

## * 유효성 검증 보고서
https://github.com/DongGeun-Yoon/nia-ssp/tree/main/doc

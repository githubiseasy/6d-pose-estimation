# 도커를 활용한 코드 실행
# - 호스트 볼륨의 경로를 사용자 환경에 맞게 수정 (절대 경로)
# - 도커 이미지명과 태그명을 적절하게 수정 (필요시)

str="docker run --name=NIA-SSP --gpus all --rm -v $PWD/data:/ssp/data -v $PWD/cfg:/ssp/cfg -v /home/user/Desktop/transparency/test_datasets:/ssp/new_dataset -v /home/user/Desktop/transparency/experimental_results:/ssp/experimental_results nia-ssp:0.1 python valid.py --datacfg data/$1/$1.data --modelcfg cfg/yolo-pose.cfg --weightfile data/$1/model/model.weights"
str2="${str} --command '${str}'"
echo $str
eval $str2
sudo yum update -y &&\
sudo yum install docker -y &&\
sudo service docker start &&\
sudo usermod -a -G docker ec2-user &&\
sudo chmod 666 /var/run/docker.sock &&\
sudo yum install tmux -y

aws configure

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 838255262149.dkr.ecr.us-east-1.amazonaws.com &&\
docker pull 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download:latest &&\
tmux new -s node-11

docker run -it --entrypoint /bin/bash 838255262149.dkr.ecr.us-east-1.amazonaws.com/tof_download 
cd src 
python3 download_and_predict_job.py --country "Senegal" --ul_flag True --year 2019
python3 fix_artifact_tile.py --country "Gambia" --ul_flag True --db_path_s3 "2020/databases/reprocess-gambia-3.csv" --db_path "/"
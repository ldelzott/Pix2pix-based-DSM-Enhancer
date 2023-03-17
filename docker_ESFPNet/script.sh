#sudo docker stop --time=0 $(sudo docker ps -a -q)
#sudo docker rm -f $(sudo docker ps -a -q)
# sudo docker rmi -f $(docker images -aq)
sudo docker build -t esfpnet . #--no-cache
sudo docker run -d --privileged -p 5000:5000 esfpnet
sudo docker ps

sudo docker stop --time=0 $(sudo docker ps -a -q)
sudo docker rm -f $(sudo docker ps -a -q)
# sudo docker rmi -f $(docker images -aq)
sudo docker build -t pix . #--no-cache
sudo docker run -d --privileged -p 6000:6000 pix
sudo docker ps

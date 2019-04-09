# Script adapted from https://github.com/akshaychawla/ImageNet-downloader
wget -c --no-check-certificate http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz
tar xzvf imagenet_fall11_urls.tgz
mv fall11_urls.txt ../data/downloads/full_imagenet_urls.txt
rm imagenet_fall11_urls.tgz
Para crear la imagen de cada uno de los modelos, desde el directorio actual:



docker build -t chatter_ia -f chatter_ia\Dockerfile .

docker build -t coder_ia -f coder_ia\Dockerfile .



Una vez se generan las imagenes de docker. Para ejecutar los contenedores que están funcionando actualmente sería:



docker run -d --gpus all --name coder_ia_container -p 7880:7880 --restart=unless-stopped  --memory="6G" --cpus="4" -d coder_ia

docker run -d --gpus all --name chatter_ia_container -p 7890:7890 --restart=unless-stopped  --memory="6G" --cpus="4" -d chatter_ia



Y ahora en localhost:7880 y localhost:7890 se deberían de mostrar las aplicaciones web.

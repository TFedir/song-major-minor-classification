docker buildx build . -t ium_modele

docker run --network host --name ium_modele ium_modele
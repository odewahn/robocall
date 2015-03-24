
## Building the image

```
$ docker build -t odewahn/robocall .
```

## Run the notebook

```
docker run -it \
   -p 8888:8888 \
   -v $(pwd):/usr/data \
   -w /usr/data \
   odewahn/robocall \
   sh -c "ipython notebook --ip=0.0.0.0 --no-browser"
```


## Concert the notebook to "Atlas" markdown

```
docker run -it \
   -v $(pwd):/usr/data \
   -w /usr/data \
   odewahn/robocall \
   ipymd --from notebook --to atlas *.ipynb
```
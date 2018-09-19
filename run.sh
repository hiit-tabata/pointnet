docker run  -it --rm \
            --runtime=nvidia \
            -v $PWD:/notebooks/ \
            -p 8888:8888 \
            -p 6006:6006 \
            -m 9216m \
            --cpus="5.5" \
            holman/pointnet:latest-gpu \
            jupyter lab --NotebookApp.token='holman' --allow-root  
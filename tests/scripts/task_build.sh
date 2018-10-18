#!/bin/bash
<<<<<<< HEAD
echo "Build TVM..."
make "$@"
cd nnvm

echo "Build NNVM..."
make "$@"
=======
cd $1 && cmake .. && make $2 && cd ..
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

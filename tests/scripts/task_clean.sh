#!/bin/bash
echo "Cleanup data..."
<<<<<<< HEAD
cd nnvm
make clean
cd ..
make clean
=======
cd $1 && rm -rf Cmake* && cd ..
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

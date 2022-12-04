python3 setup.py build_ext --inplace --build-lib=demons;
rm -rf ../demons;
mv demons ..;
rm -f **/*.so **/*.c **/*.o;

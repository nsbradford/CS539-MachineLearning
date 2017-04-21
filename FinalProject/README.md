# CNN Batch Normalization

### 1. Download data

Fire up your terminal and run the following:

```bash
> wget --continue https://archive.org/download/comma-dataset/comma-dataset.zip
```

To unzip a single pair of files without unzipping the whole dataset (~80GB), pick a filename pair (camera and log):

```bash
> unzip -l comma-dataset.zip  # view files insize zip 
Archive:  comma-dataset.zip
  Length      Date    Time    Name
----------  ---------- -----   ----
       387  08-01-2016 01:14   LICENSE
         0  08-01-2016 01:22   camera/
8178897856  08-01-2016 01:17   camera/2016-01-30--11-24-51.h5
9122616256  08-01-2016 00:22   camera/2016-01-30--13-46-00.h5
3145733056  08-01-2016 00:24   camera/2016-01-31--19-19-25.h5
8650757056  08-01-2016 00:27   camera/2016-02-02--10-16-58.h5
...
        0  08-01-2016 00:48   log/
309976614  08-01-2016 00:47   log/2016-01-30--11-24-51.h5
346214946  08-01-2016 00:47   log/2016-01-30--13-46-00.h5
118611986  08-01-2016 00:47   log/2016-01-31--19-19-25.h5
333273584  08-01-2016 00:47   log/2016-02-02--10-16-58.h5
...
```

Then unzip them into the `dataset` directory.

```bash
> mkdir -p dataset
> cd dataset
> unzip ../comma-dataset.zip camera/2016-01-30--11-24-51.h5 log/2016-01-30--11-24-51.h5
```

## How to run demo on arbitray online images
### Install yolov7 and Implicit3DUnderstanding
clone these two repository firstly, and install their required packages.

### Download all required pretrained weights.
The required pretrained weights should be arranged as follows:
```
checkpoints
│   yolov7.pt   
│
└───im3d_weight
│      model_best.pth
│      out_config.yaml
│   
└───instPIFu
       model_best_pix3d.pth
```
<br>
Download <a href="https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt">yolov7<a/> and put it under ./checkpoints.<br>
Download <a href="">Im3D weight</a>, extract it and put it under ./checkpoints, renamed the folder as im3d_weight. <br>
Download <a href="https://cuhko365-my.sharepoint.com/:u:/g/personal/115010192_link_cuhk_edu_cn/ES4SqMFhnR9DipjSWhBt5C4BomRDF7jO-7AE1v-FaS5l6g?e=V38qt0">model_best_pix3d.pth</a>, and put it under checkpoints/instPIFu.

### run the demo
Before running the demo, you need to download <a href="https://cuhko365-my.sharepoint.com/:u:/g/personal/115010192_link_cuhk_edu_cn/ES3sTytzWkZFtcHLHieA98YBiCmbDDCjdYofIDlr7mj1QA?e=rYLJ2A">sunrgbd_train_test_data</a>, and put it in somewhere.
Then, modify the several path entries in run_demo.sh, make sure they are pointing to the right path.
```angular2html
bash run_demo.sh
```
Results will be saved at InstPIFu/real_demo/taskid



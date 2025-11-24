'''
Documentation for the KITTI-dataset:
@article{Geiger2013IJRR,
  author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
  title = {Vision meets Robotics: The KITTI Dataset},
  journal = {International Journal of Robotics Research (IJRR)},
  year = {2013}
}
'''

# bsub < HPCtrain.sh            Bash script to run this file on a cluster
# bstat                         Check job status on cluster

from ultralytics import YOLO

model = YOLO("yolo11n.yaml").load("yolo11n.pt")

train_results = model.train(
    data="kitti.yaml",
    epochs=100,
    imgsz=640,
    project="kitti_yolo_runs",
    name="exp1",
    save=True,
)

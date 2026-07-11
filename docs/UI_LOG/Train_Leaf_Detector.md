Train Leaf Detector
src/training/train_yolo_leaf_detector.py • Logs: latest-only • Runtime: 4m 48s • Exit: 0
completed
Progress: 100.0% • Stage: pending
Total ETA: n/a
Started: /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/.venv/bin/python3 /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/src/training/train_yolo_leaf_detector.py
[*] Generating auto-labeled YOLO dataset...
[*] Scanning images in /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/dataset/train...
[*] Generating leaf annotations for 2000 samples...
Processed 200/2000...
Processed 400/2000...
Processed 600/2000...
Processed 800/2000...
Processed 1000/2000...
Processed 1200/2000...
Processed 1400/2000...
Processed 1600/2000...
Processed 1800/2000...
Processed 2000/2000...
[+] Successfully generated YOLO dataset under /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/dataset/yolo_dataset with 2000 images.
[*] Initializing YOLO26m pre-trained model...
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 0% ──────────── 86.4KB/42.2MB 150.7KB/s 0.2s<4:46
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 0% ──────────── 129.7KB/42.2MB 218.2KB/s 0.3s<3:17
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 0% ──────────── 216.1KB/42.2MB 265.7KB/s 0.5s<2:42
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 0% ──────────── 302.5KB/42.2MB 356.3KB/s 0.7s<2:00
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 0% ──────────── 389.0KB/42.2MB 406.8KB/s 0.8s<1:45
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 0% ──────────── 432.2KB/42.2MB 413.2KB/s 0.9s<1:44
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 1% ──────────── 518.6KB/42.2MB 439.7KB/s 1.1s<1:37
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 1% ──────────── 561.8KB/42.2MB 435.9KB/s 1.2s<1:38
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 1% ──────────── 648.3KB/42.2MB 472.2KB/s 1.4s<1:30
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 1% ──────────── 734.7KB/42.2MB 473.4KB/s 1.5s<1:30
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 1% ──────────── 821.1KB/42.2MB 494.3KB/s 1.7s<1:26
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 2% ──────────── 907.6KB/42.2MB 521.9KB/s 1.9s<1:21
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 2% ──────────── 994.0KB/42.2MB 503.1KB/s 2.0s<1:24
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 2% ──────────── 1.1/42.2MB 525.2KB/s 2.2s<1:20
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 2% ──────────── 1.1/42.2MB 496.6KB/s 2.3s<1:25
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 2% ──────────── 1.2/42.2MB 516.2KB/s 2.4s<1:21
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 2% ──────────── 1.3/42.2MB 437.1KB/s 2.8s<1:36
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 3% ──────────── 1.4/42.2MB 1.1MB/s 2.9s<38.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 3% ──────────── 1.5/42.2MB 480.1KB/s 3.1s<1:27
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 3% ──────────── 1.6/42.2MB 474.1KB/s 3.2s<1:28
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 3% ──────────── 1.6/42.2MB 435.7KB/s 3.5s<1:35
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 4% ╸─────────── 1.8/42.2MB 526.4KB/s 3.7s<1:19
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 4% ╸─────────── 1.9/42.2MB 535.6KB/s 3.8s<1:17
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 4% ╸─────────── 1.9/42.2MB 511.5KB/s 4.0s<1:21
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 4% ╸─────────── 2.0/42.2MB 529.1KB/s 4.2s<1:18
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 4% ╸─────────── 2.1/42.2MB 496.4KB/s 4.3s<1:23
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 5% ╸─────────── 2.2/42.2MB 514.8KB/s 4.4s<1:20
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 5% ╸─────────── 2.2/42.2MB 486.3KB/s 4.5s<1:24
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 5% ╸─────────── 2.3/42.2MB 498.4KB/s 4.7s<1:22
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 5% ╸─────────── 2.4/42.2MB 516.9KB/s 4.8s<1:19
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 5% ╸─────────── 2.5/42.2MB 1.4MB/s 5.0s<27.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 6% ╸─────────── 2.7/42.2MB 1.8MB/s 5.1s<21.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 6% ╸─────────── 3.0/42.2MB 1.9MB/s 5.2s<20.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 7% ╸─────────── 3.2/42.2MB 2.0MB/s 5.3s<19.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 7% ╸─────────── 3.4/42.2MB 2.0MB/s 5.4s<19.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 8% ━─────────── 3.6/42.2MB 2.2MB/s 5.5s<17.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 9% ━─────────── 3.8/42.2MB 2.1MB/s 5.6s<18.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 9% ━─────────── 4.1/42.2MB 2.2MB/s 5.7s<17.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 10% ━─────────── 4.3/42.2MB 1.3MB/s 5.9s<29.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 10% ━─────────── 4.3/42.2MB 470.6KB/s 6.0s<1:22
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 10% ━─────────── 4.4/42.2MB 481.5KB/s 6.2s<1:20
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 10% ━─────────── 4.5/42.2MB 458.1KB/s 6.3s<1:24
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 10% ━─────────── 4.5/42.2MB 441.5KB/s 6.4s<1:27
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 10% ━─────────── 4.6/42.2MB 461.5KB/s 6.6s<1:23
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 10% ━─────────── 4.6/42.2MB 439.7KB/s 6.7s<1:27
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 11% ━─────────── 4.7/42.2MB 461.7KB/s 6.8s<1:23
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 11% ━─────────── 4.8/42.2MB 451.8KB/s 6.9s<1:25
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 11% ━─────────── 4.9/42.2MB 476.8KB/s 7.1s<1:20
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 11% ━─────────── 4.9/42.2MB 461.9KB/s 7.2s<1:23
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 11% ━─────────── 5.0/42.2MB 479.6KB/s 7.4s<1:19
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 11% ━─────────── 5.0/42.2MB 401.1KB/s 7.6s<1:35
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 12% ━─────────── 5.1/42.2MB 534.8KB/s 7.7s<1:11
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 12% ━─────────── 5.1/42.2MB 500.4KB/s 7.8s<1:16
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 12% ━─────────── 5.2/42.2MB 479.1KB/s 8.0s<1:19
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 12% ━╸────────── 5.3/42.2MB 490.0KB/s 8.1s<1:17
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 12% ━╸────────── 5.4/42.2MB 490.5KB/s 8.3s<1:17
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 12% ━╸────────── 5.4/42.2MB 392.2KB/s 8.6s<1:36
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 13% ━╸────────── 5.6/42.2MB 555.3KB/s 8.7s<1:08
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 13% ━╸────────── 5.6/42.2MB 512.1KB/s 8.8s<1:13
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 13% ━╸────────── 5.7/42.2MB 484.1KB/s 8.9s<1:17
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 13% ━╸────────── 5.7/42.2MB 484.1KB/s 9.1s<1:17
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 13% ━╸────────── 5.8/42.2MB 464.7KB/s 9.2s<1:20
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 13% ━╸────────── 5.9/42.2MB 475.2KB/s 9.4s<1:18
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 14% ━╸────────── 6.0/42.2MB 488.3KB/s 9.6s<1:16
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 14% ━╸────────── 6.0/42.2MB 462.7KB/s 9.7s<1:20
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 14% ━╸────────── 6.0/42.2MB 448.9KB/s 9.8s<1:23
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 14% ━╸────────── 6.1/42.2MB 549.5KB/s 9.9s<1:07
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 14% ━╸────────── 6.3/42.2MB 1.8MB/s 10.0s<20.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 15% ━╸────────── 6.5/42.2MB 2.0MB/s 10.1s<18.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 15% ━╸────────── 6.8/42.2MB 2.0MB/s 10.2s<17.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 16% ━╸────────── 7.0/42.2MB 2.1MB/s 10.3s<16.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 17% ━━────────── 7.3/42.2MB 2.3MB/s 10.4s<15.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 17% ━━────────── 7.5/42.2MB 2.4MB/s 10.5s<14.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 18% ━━────────── 7.8/42.2MB 2.2MB/s 10.7s<15.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 18% ━━────────── 8.0/42.2MB 2.4MB/s 10.8s<14.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 19% ━━────────── 8.2/42.2MB 663.3KB/s 10.9s<52.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 19% ━━────────── 8.3/42.2MB 622.6KB/s 11.1s<55.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 19% ━━────────── 8.3/42.2MB 561.7KB/s 11.2s<1:02
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 19% ━━────────── 8.4/42.2MB 545.9KB/s 11.4s<1:03
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 20% ━━────────── 8.5/42.2MB 510.9KB/s 11.6s<1:08
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 20% ━━────────── 8.6/42.2MB 522.2KB/s 11.7s<1:06
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 20% ━━────────── 8.7/42.2MB 452.1KB/s 12.0s<1:16
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 20% ━━────────── 8.8/42.2MB 567.6KB/s 12.2s<1:00
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 20% ━━╸───────── 8.9/42.2MB 536.7KB/s 12.4s<1:04
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 21% ━━╸───────── 8.9/42.2MB 501.8KB/s 12.5s<1:08
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 21% ━━╸───────── 9.0/42.2MB 510.6KB/s 12.6s<1:07
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 21% ━━╸───────── 9.1/42.2MB 512.6KB/s 12.8s<1:06
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 21% ━━╸───────── 9.2/42.2MB 490.3KB/s 13.0s<1:09
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 21% ━━╸───────── 9.2/42.2MB 497.9KB/s 13.2s<1:08
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 21% ━━╸───────── 9.3/42.2MB 467.7KB/s 13.3s<1:12
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 22% ━━╸───────── 9.4/42.2MB 487.6KB/s 13.5s<1:09
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 22% ━━╸───────── 9.4/42.2MB 464.6KB/s 13.6s<1:12
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 22% ━━╸───────── 9.5/42.2MB 451.9KB/s 13.7s<1:14
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 22% ━━╸───────── 9.5/42.2MB 475.3KB/s 13.8s<1:10
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 22% ━━╸───────── 9.6/42.2MB 449.2KB/s 13.9s<1:14
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 22% ━━╸───────── 9.7/42.2MB 458.9KB/s 14.1s<1:13
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 22% ━━╸───────── 9.7/42.2MB 449.7KB/s 14.2s<1:14
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 23% ━━╸───────── 9.7/42.2MB 438.3KB/s 14.3s<1:16
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 23% ━━╸───────── 9.8/42.2MB 451.8KB/s 14.5s<1:13
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 23% ━━╸───────── 9.9/42.2MB 435.5KB/s 14.6s<1:16
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 23% ━━╸───────── 10.0/42.2MB 444.9KB/s 14.8s<1:14
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 23% ━━╸───────── 10.1/42.2MB 1020.3KB/s 14.9s<32.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 24% ━━╸───────── 10.4/42.2MB 2.7MB/s 15.0s<11.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 25% ━━━───────── 10.7/42.2MB 2.7MB/s 15.1s<11.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 25% ━━━───────── 11.0/42.2MB 2.9MB/s 15.2s<10.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 26% ━━━───────── 11.3/42.2MB 3.0MB/s 15.4s<10.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 27% ━━━───────── 11.6/42.2MB 3.2MB/s 15.5s<9.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 28% ━━━───────── 12.0/42.2MB 3.0MB/s 15.6s<10.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 29% ━━━╸──────── 12.4/42.2MB 3.2MB/s 15.7s<9.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 30% ━━━╸──────── 12.7/42.2MB 3.3MB/s 15.8s<8.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 30% ━━━╸──────── 12.8/42.2MB 541.5KB/s 16.0s<55.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 30% ━━━╸──────── 12.9/42.2MB 463.7KB/s 16.1s<1:05
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 30% ━━━╸──────── 13.0/42.2MB 478.9KB/s 16.3s<1:03
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 30% ━━━╸──────── 13.0/42.2MB 458.8KB/s 16.4s<1:05
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 30% ━━━╸──────── 13.1/42.2MB 460.1KB/s 16.6s<1:05
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 31% ━━━╸──────── 13.1/42.2MB 450.3KB/s 16.7s<1:06
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 31% ━━━╸──────── 13.2/42.2MB 374.7KB/s 17.1s<1:19
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 31% ━━━╸──────── 13.4/42.2MB 521.7KB/s 17.3s<56.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 31% ━━━╸──────── 13.4/42.2MB 477.4KB/s 17.4s<1:02
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 31% ━━━╸──────── 13.5/42.2MB 475.9KB/s 17.6s<1:02
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 32% ━━━╸──────── 13.5/42.2MB 442.3KB/s 17.7s<1:06
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 32% ━━━╸──────── 13.6/42.2MB 448.8KB/s 17.9s<1:05
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 32% ━━━╸──────── 13.7/42.2MB 433.0KB/s 18.0s<1:07
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 32% ━━━╸──────── 13.7/42.2MB 413.7KB/s 18.1s<1:11
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 32% ━━━╸──────── 13.8/42.2MB 423.1KB/s 18.3s<1:09
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 32% ━━━╸──────── 13.8/42.2MB 421.1KB/s 18.4s<1:09
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 32% ━━━╸──────── 13.9/42.2MB 365.5KB/s 18.8s<1:19
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 33% ━━━╸──────── 14.1/42.2MB 475.7KB/s 19.0s<1:01
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 33% ━━━━──────── 14.1/42.2MB 439.6KB/s 19.1s<1:05
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 33% ━━━━──────── 14.2/42.2MB 455.5KB/s 19.3s<1:03
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 33% ━━━━──────── 14.2/42.2MB 429.7KB/s 19.4s<1:07
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 33% ━━━━──────── 14.3/42.2MB 410.4KB/s 19.5s<1:10
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 33% ━━━━──────── 14.3/42.2MB 348.0KB/s 19.7s<1:22
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 34% ━━━━──────── 14.4/42.2MB 481.2KB/s 19.8s<59.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 34% ━━━━──────── 14.5/42.2MB 1.1MB/s 20.0s<25.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 34% ━━━━──────── 14.7/42.2MB 1.8MB/s 20.1s<15.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 35% ━━━━──────── 14.9/42.2MB 1.8MB/s 20.2s<15.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 35% ━━━━──────── 15.2/42.2MB 2.0MB/s 20.3s<13.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 36% ━━━━──────── 15.4/42.2MB 2.0MB/s 20.4s<13.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 36% ━━━━──────── 15.6/42.2MB 2.1MB/s 20.5s<12.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 37% ━━━━╸─────── 15.9/42.2MB 2.3MB/s 20.6s<11.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 38% ━━━━╸─────── 16.1/42.2MB 2.2MB/s 20.7s<11.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 38% ━━━━╸─────── 16.3/42.2MB 1.2MB/s 20.9s<22.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 38% ━━━━╸─────── 16.3/42.2MB 445.9KB/s 21.0s<59.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 38% ━━━━╸─────── 16.4/42.2MB 441.5KB/s 21.1s<59.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 38% ━━━━╸─────── 16.5/42.2MB 459.1KB/s 21.3s<57.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 39% ━━━━╸─────── 16.5/42.2MB 445.6KB/s 21.4s<59.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 39% ━━━━╸─────── 16.5/42.2MB 438.5KB/s 21.5s<59.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 39% ━━━━╸─────── 16.6/42.2MB 356.0KB/s 21.8s<1:14
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 39% ━━━━╸─────── 16.7/42.2MB 1.1MB/s 21.9s<23.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 39% ━━━━╸─────── 16.8/42.2MB 375.0KB/s 22.0s<1:09
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 39% ━━━━╸─────── 16.8/42.2MB 417.4KB/s 22.1s<1:02
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 39% ━━━━╸─────── 16.9/42.2MB 402.8KB/s 22.3s<1:04
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 40% ━━━━╸─────── 16.9/42.2MB 390.9KB/s 22.4s<1:06
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 40% ━━━━╸─────── 17.0/42.2MB 390.6KB/s 22.5s<1:06
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 40% ━━━━╸─────── 17.0/42.2MB 394.8KB/s 22.6s<1:05
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 40% ━━━━╸─────── 17.1/42.2MB 384.3KB/s 22.8s<1:07
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 40% ━━━━╸─────── 17.2/42.2MB 390.9KB/s 23.0s<1:06
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 40% ━━━━╸─────── 17.2/42.2MB 381.9KB/s 23.2s<1:07
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 40% ━━━━╸─────── 17.3/42.2MB 396.6KB/s 23.4s<1:04
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 41% ━━━━╸─────── 17.3/42.2MB 383.0KB/s 23.5s<1:06
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 41% ━━━━╸─────── 17.4/42.2MB 377.4KB/s 23.6s<1:07
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 41% ━━━━╸─────── 17.5/42.2MB 399.4KB/s 23.8s<1:03
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 41% ━━━━╸─────── 17.5/42.2MB 371.8KB/s 23.9s<1:08
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 41% ━━━━━─────── 17.6/42.2MB 397.3KB/s 24.1s<1:03
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 41% ━━━━━─────── 17.6/42.2MB 376.2KB/s 24.3s<1:07
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 41% ━━━━━─────── 17.7/42.2MB 397.3KB/s 24.5s<1:03
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 42% ━━━━━─────── 17.8/42.2MB 385.5KB/s 24.6s<1:05
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 42% ━━━━━─────── 17.8/42.2MB 380.8KB/s 24.7s<1:06
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 42% ━━━━━─────── 17.9/42.2MB 423.7KB/s 24.9s<58.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 42% ━━━━━─────── 18.0/42.2MB 1.2MB/s 25.0s<20.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 43% ━━━━━─────── 18.3/42.2MB 2.4MB/s 25.1s<9.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 43% ━━━━━─────── 18.6/42.2MB 2.6MB/s 25.2s<9.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 44% ━━━━━─────── 18.9/42.2MB 2.7MB/s 25.3s<8.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 45% ━━━━━─────── 19.2/42.2MB 2.9MB/s 25.4s<7.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 46% ━━━━━╸────── 19.5/42.2MB 2.9MB/s 25.5s<8.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 46% ━━━━━╸────── 19.8/42.2MB 3.0MB/s 25.6s<7.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 47% ━━━━━╸────── 20.1/42.2MB 3.3MB/s 25.7s<6.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 48% ━━━━━╸────── 20.4/42.2MB 2.1MB/s 25.9s<10.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 48% ━━━━━╸────── 20.5/42.2MB 424.0KB/s 26.1s<52.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 48% ━━━━━╸────── 20.6/42.2MB 420.3KB/s 26.3s<52.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 48% ━━━━━╸────── 20.7/42.2MB 438.3KB/s 26.5s<50.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 49% ━━━━━╸────── 20.7/42.2MB 414.5KB/s 26.6s<53.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 49% ━━━━━╸────── 20.8/42.2MB 408.9KB/s 26.7s<53.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 49% ━━━━━╸────── 20.8/42.2MB 425.2KB/s 26.9s<51.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 49% ━━━━━╸────── 20.9/42.2MB 410.2KB/s 27.0s<53.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 49% ━━━━━╸────── 21.0/42.2MB 427.1KB/s 27.2s<50.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 49% ━━━━━╸────── 21.0/42.2MB 406.0KB/s 27.3s<53.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 49% ━━━━━╸────── 21.1/42.2MB 423.6KB/s 27.5s<51.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 50% ━━━━━━────── 21.1/42.2MB 418.5KB/s 27.6s<51.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 50% ━━━━━━────── 21.2/42.2MB 407.5KB/s 27.7s<52.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 50% ━━━━━━────── 21.3/42.2MB 424.3KB/s 27.9s<50.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 50% ━━━━━━────── 21.3/42.2MB 420.7KB/s 28.0s<50.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 50% ━━━━━━────── 21.4/42.2MB 445.4KB/s 28.2s<47.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 50% ━━━━━━────── 21.4/42.2MB 417.4KB/s 28.3s<50.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 50% ━━━━━━────── 21.5/42.2MB 435.0KB/s 28.5s<48.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 51% ━━━━━━────── 21.6/42.2MB 414.8KB/s 28.6s<51.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 51% ━━━━━━────── 21.6/42.2MB 411.7KB/s 28.7s<51.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 51% ━━━━━━────── 21.7/42.2MB 347.2KB/s 29.1s<1:00
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 51% ━━━━━━────── 21.9/42.2MB 494.5KB/s 29.3s<42.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 51% ━━━━━━────── 21.9/42.2MB 484.8KB/s 29.5s<42.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 52% ━━━━━━────── 22.0/42.2MB 466.8KB/s 29.7s<44.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 52% ━━━━━━────── 22.1/42.2MB 487.7KB/s 29.9s<42.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 52% ━━━━━━────── 22.3/42.2MB 1.9MB/s 30.0s<10.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 53% ━━━━━━────── 22.7/42.2MB 3.0MB/s 30.1s<6.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 54% ━━━━━━╸───── 23.0/42.2MB 2.9MB/s 30.2s<6.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 55% ━━━━━━╸───── 23.3/42.2MB 3.1MB/s 30.3s<6.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 56% ━━━━━━╸───── 23.7/42.2MB 3.4MB/s 30.4s<5.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 56% ━━━━━━╸───── 23.8/42.2MB 1.5MB/s 30.5s<12.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 57% ━━━━━━╸───── 24.3/42.2MB 4.1MB/s 30.7s<4.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 58% ━━━━━━╸───── 24.6/42.2MB 2.3MB/s 30.8s<7.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 58% ━━━━━━━───── 24.7/42.2MB 1.1MB/s 30.9s<16.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 58% ━━━━━━━───── 24.8/42.2MB 484.5KB/s 31.1s<36.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 58% ━━━━━━━───── 24.8/42.2MB 455.6KB/s 31.2s<39.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 58% ━━━━━━━───── 24.9/42.2MB 444.3KB/s 31.3s<40.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 59% ━━━━━━━───── 24.9/42.2MB 458.9KB/s 31.5s<38.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 59% ━━━━━━━───── 25.0/42.2MB 435.0KB/s 31.6s<40.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 59% ━━━━━━━───── 25.1/42.2MB 456.1KB/s 31.7s<38.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 59% ━━━━━━━───── 25.1/42.2MB 430.3KB/s 31.9s<40.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 59% ━━━━━━━───── 25.2/42.2MB 440.6KB/s 32.0s<39.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 59% ━━━━━━━───── 25.2/42.2MB 425.6KB/s 32.2s<40.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 59% ━━━━━━━───── 25.3/42.2MB 399.9KB/s 32.3s<43.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 60% ━━━━━━━───── 25.4/42.2MB 411.7KB/s 32.5s<41.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 60% ━━━━━━━───── 25.4/42.2MB 401.4KB/s 32.6s<42.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 60% ━━━━━━━───── 25.5/42.2MB 410.6KB/s 32.8s<41.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 60% ━━━━━━━───── 25.5/42.2MB 395.1KB/s 32.9s<43.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 60% ━━━━━━━───── 25.6/42.2MB 407.3KB/s 33.1s<41.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 60% ━━━━━━━───── 25.7/42.2MB 397.0KB/s 33.2s<42.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 60% ━━━━━━━───── 25.7/42.2MB 387.1KB/s 33.3s<43.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 61% ━━━━━━━───── 25.8/42.2MB 398.2KB/s 33.5s<42.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 61% ━━━━━━━───── 25.8/42.2MB 395.0KB/s 33.7s<42.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 61% ━━━━━━━───── 25.9/42.2MB 313.7KB/s 34.0s<53.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 61% ━━━━━━━───── 26.0/42.2MB 1.1MB/s 34.1s<15.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 61% ━━━━━━━───── 26.0/42.2MB 312.7KB/s 34.3s<52.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 61% ━━━━━━━───── 26.1/42.2MB 324.6KB/s 34.4s<50.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 61% ━━━━━━━───── 26.1/42.2MB 340.7KB/s 34.5s<48.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 62% ━━━━━━━───── 26.2/42.2MB 370.3KB/s 34.7s<44.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 62% ━━━━━━━───── 26.3/42.2MB 376.0KB/s 34.8s<43.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 62% ━━━━━━━───── 26.3/42.2MB 479.7KB/s 34.9s<33.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 62% ━━━━━━━╸──── 26.6/42.2MB 2.5MB/s 35.0s<6.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 63% ━━━━━━━╸──── 26.9/42.2MB 2.8MB/s 35.1s<5.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 64% ━━━━━━━╸──── 27.2/42.2MB 3.0MB/s 35.2s<4.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 65% ━━━━━━━╸──── 27.5/42.2MB 2.0MB/s 35.4s<7.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 66% ━━━━━━━╸──── 27.9/42.2MB 3.3MB/s 35.5s<4.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 66% ━━━━━━━╸──── 28.1/42.2MB 2.1MB/s 35.6s<6.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 67% ━━━━━━━━──── 28.4/42.2MB 2.5MB/s 35.7s<5.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 67% ━━━━━━━━──── 28.5/42.2MB 622.8KB/s 35.9s<22.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 67% ━━━━━━━━──── 28.6/42.2MB 570.5KB/s 36.1s<24.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 67% ━━━━━━━━──── 28.7/42.2MB 510.5KB/s 36.2s<27.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 68% ━━━━━━━━──── 28.7/42.2MB 499.4KB/s 36.4s<27.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 68% ━━━━━━━━──── 28.8/42.2MB 473.5KB/s 36.5s<29.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 68% ━━━━━━━━──── 28.9/42.2MB 447.5KB/s 36.7s<30.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 68% ━━━━━━━━──── 28.9/42.2MB 434.6KB/s 36.8s<31.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 68% ━━━━━━━━──── 29.0/42.2MB 423.1KB/s 36.9s<32.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 68% ━━━━━━━━──── 29.0/42.2MB 439.1KB/s 37.1s<30.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 68% ━━━━━━━━──── 29.1/42.2MB 342.9KB/s 37.5s<39.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 69% ━━━━━━━━──── 29.2/42.2MB 1.2MB/s 37.6s<11.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 69% ━━━━━━━━──── 29.3/42.2MB 369.2KB/s 37.8s<35.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 69% ━━━━━━━━──── 29.3/42.2MB 382.4KB/s 37.9s<34.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 69% ━━━━━━━━──── 29.4/42.2MB 312.4KB/s 38.2s<42.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 69% ━━━━━━━━──── 29.5/42.2MB 476.9KB/s 38.3s<27.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 70% ━━━━━━━━──── 29.6/42.2MB 480.7KB/s 38.5s<26.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 70% ━━━━━━━━──── 29.6/42.2MB 453.7KB/s 38.6s<28.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 70% ━━━━━━━━──── 29.7/42.2MB 460.8KB/s 38.8s<27.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 70% ━━━━━━━━──── 29.8/42.2MB 443.6KB/s 38.9s<28.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 70% ━━━━━━━━──── 29.8/42.2MB 431.8KB/s 39.0s<29.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 70% ━━━━━━━━──── 29.9/42.2MB 446.7KB/s 39.2s<28.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 70% ━━━━━━━━╸─── 29.9/42.2MB 438.5KB/s 39.3s<28.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 71% ━━━━━━━━╸─── 30.0/42.2MB 443.5KB/s 39.5s<28.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 71% ━━━━━━━━╸─── 30.0/42.2MB 438.3KB/s 39.6s<28.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 71% ━━━━━━━━╸─── 30.1/42.2MB 411.3KB/s 39.8s<30.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 71% ━━━━━━━━╸─── 30.3/42.2MB 1.6MB/s 39.9s<7.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 72% ━━━━━━━━╸─── 30.5/42.2MB 1.4MB/s 40.1s<8.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 72% ━━━━━━━━╸─── 30.6/42.2MB 1.5MB/s 40.2s<8.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 72% ━━━━━━━━╸─── 30.8/42.2MB 1.5MB/s 40.3s<7.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 73% ━━━━━━━━╸─── 31.0/42.2MB 1.7MB/s 40.4s<6.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 73% ━━━━━━━━╸─── 31.1/42.2MB 1016.6KB/s 40.5s<11.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 74% ━━━━━━━━╸─── 31.4/42.2MB 2.0MB/s 40.6s<5.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 74% ━━━━━━━━╸─── 31.5/42.2MB 1.4MB/s 40.8s<7.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 74% ━━━━━━━━╸─── 31.6/42.2MB 433.2KB/s 40.9s<25.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 74% ━━━━━━━━╸─── 31.7/42.2MB 414.2KB/s 41.1s<26.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 75% ━━━━━━━━━─── 31.7/42.2MB 421.9KB/s 41.3s<25.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 75% ━━━━━━━━━─── 31.8/42.2MB 410.2KB/s 41.4s<26.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 75% ━━━━━━━━━─── 31.9/42.2MB 424.9KB/s 41.6s<24.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 75% ━━━━━━━━━─── 31.9/42.2MB 411.8KB/s 41.7s<25.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 75% ━━━━━━━━━─── 32.0/42.2MB 420.3KB/s 41.9s<24.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 75% ━━━━━━━━━─── 32.0/42.2MB 405.2KB/s 42.0s<25.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 75% ━━━━━━━━━─── 32.1/42.2MB 391.0KB/s 42.1s<26.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 76% ━━━━━━━━━─── 32.2/42.2MB 411.6KB/s 42.3s<25.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 76% ━━━━━━━━━─── 32.2/42.2MB 399.9KB/s 42.4s<25.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 76% ━━━━━━━━━─── 32.3/42.2MB 411.1KB/s 42.6s<24.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 76% ━━━━━━━━━─── 32.3/42.2MB 400.7KB/s 42.7s<25.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 76% ━━━━━━━━━─── 32.4/42.2MB 416.7KB/s 42.9s<24.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 76% ━━━━━━━━━─── 32.5/42.2MB 400.5KB/s 43.0s<24.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 76% ━━━━━━━━━─── 32.5/42.2MB 392.8KB/s 43.1s<25.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 77% ━━━━━━━━━─── 32.6/42.2MB 408.6KB/s 43.3s<24.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 77% ━━━━━━━━━─── 32.6/42.2MB 397.3KB/s 43.5s<24.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 77% ━━━━━━━━━─── 32.7/42.2MB 415.7KB/s 43.6s<23.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 77% ━━━━━━━━━─── 32.8/42.2MB 398.3KB/s 43.8s<24.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 77% ━━━━━━━━━─── 32.8/42.2MB 414.6KB/s 44.0s<23.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 77% ━━━━━━━━━─── 32.9/42.2MB 407.2KB/s 44.1s<23.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 77% ━━━━━━━━━─── 32.9/42.2MB 408.8KB/s 44.2s<23.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 78% ━━━━━━━━━─── 33.0/42.2MB 424.1KB/s 44.4s<22.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 78% ━━━━━━━━━─── 33.0/42.2MB 402.3KB/s 44.5s<23.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 78% ━━━━━━━━━─── 33.1/42.2MB 420.0KB/s 44.7s<22.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 78% ━━━━━━━━━─── 33.2/42.2MB 405.4KB/s 44.8s<22.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 78% ━━━━━━━━━─── 33.3/42.2MB 1.0MB/s 44.9s<8.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 79% ━━━━━━━━━╸── 33.6/42.2MB 2.2MB/s 45.0s<3.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 79% ━━━━━━━━━╸── 33.8/42.2MB 2.1MB/s 45.1s<4.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 80% ━━━━━━━━━╸── 34.0/42.2MB 2.3MB/s 45.2s<3.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 81% ━━━━━━━━━╸── 34.3/42.2MB 2.4MB/s 45.3s<3.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 81% ━━━━━━━━━╸── 34.6/42.2MB 2.5MB/s 45.5s<3.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 82% ━━━━━━━━━╸── 34.9/42.2MB 2.6MB/s 45.6s<2.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 83% ━━━━━━━━━╸── 35.2/42.2MB 2.6MB/s 45.7s<2.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 83% ━━━━━━━━━━── 35.5/42.2MB 2.8MB/s 45.8s<2.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 84% ━━━━━━━━━━── 35.5/42.2MB 515.6KB/s 45.9s<13.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 84% ━━━━━━━━━━── 35.6/42.2MB 460.2KB/s 46.0s<14.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 84% ━━━━━━━━━━── 35.7/42.2MB 473.0KB/s 46.2s<14.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 84% ━━━━━━━━━━── 35.7/42.2MB 456.8KB/s 46.3s<14.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 84% ━━━━━━━━━━── 35.7/42.2MB 429.6KB/s 46.4s<15.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 84% ━━━━━━━━━━── 35.8/42.2MB 436.2KB/s 46.6s<15.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 84% ━━━━━━━━━━── 35.9/42.2MB 424.1KB/s 46.7s<15.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 85% ━━━━━━━━━━── 36.0/42.2MB 433.4KB/s 46.9s<14.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 85% ━━━━━━━━━━── 36.0/42.2MB 448.6KB/s 47.1s<14.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 85% ━━━━━━━━━━── 36.1/42.2MB 435.3KB/s 47.2s<14.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 85% ━━━━━━━━━━── 36.1/42.2MB 410.1KB/s 47.3s<15.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 85% ━━━━━━━━━━── 36.2/42.2MB 407.3KB/s 47.4s<15.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 85% ━━━━━━━━━━── 36.2/42.2MB 318.2KB/s 47.8s<19.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 86% ━━━━━━━━━━── 36.4/42.2MB 1010.1KB/s 48.0s<5.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 86% ━━━━━━━━━━── 36.4/42.2MB 342.3KB/s 48.1s<17.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 86% ━━━━━━━━━━── 36.5/42.2MB 377.6KB/s 48.3s<15.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 86% ━━━━━━━━━━── 36.5/42.2MB 381.1KB/s 48.4s<15.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 86% ━━━━━━━━━━── 36.6/42.2MB 390.3KB/s 48.5s<14.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 86% ━━━━━━━━━━── 36.7/42.2MB 415.0KB/s 48.7s<13.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 86% ━━━━━━━━━━── 36.7/42.2MB 407.9KB/s 48.8s<13.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 87% ━━━━━━━━━━── 36.8/42.2MB 432.9KB/s 49.0s<12.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 87% ━━━━━━━━━━── 36.8/42.2MB 422.4KB/s 49.1s<13.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 87% ━━━━━━━━━━── 36.9/42.2MB 443.9KB/s 49.3s<12.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 87% ━━━━━━━━━━╸─ 37.0/42.2MB 350.6KB/s 49.6s<15.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 87% ━━━━━━━━━━╸─ 37.1/42.2MB 488.8KB/s 49.7s<10.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 87% ━━━━━━━━━━╸─ 37.1/42.2MB 468.2KB/s 49.9s<11.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 88% ━━━━━━━━━━╸─ 37.3/42.2MB 1.6MB/s 50.0s<3.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 88% ━━━━━━━━━━╸─ 37.5/42.2MB 2.1MB/s 50.1s<2.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 89% ━━━━━━━━━━╸─ 37.7/42.2MB 1.3MB/s 50.2s<3.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 89% ━━━━━━━━━━╸─ 38.0/42.2MB 2.4MB/s 50.3s<1.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 90% ━━━━━━━━━━╸─ 38.2/42.2MB 1.6MB/s 50.4s<2.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 90% ━━━━━━━━━━╸─ 38.4/42.2MB 1.8MB/s 50.5s<2.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 91% ━━━━━━━━━━╸─ 38.6/42.2MB 1.8MB/s 50.7s<2.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 91% ━━━━━━━━━━━─ 38.8/42.2MB 1.9MB/s 50.8s<1.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 92% ━━━━━━━━━━━─ 38.9/42.2MB 518.8KB/s 51.0s<6.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 92% ━━━━━━━━━━━─ 39.0/42.2MB 479.2KB/s 51.1s<6.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 92% ━━━━━━━━━━━─ 39.0/42.2MB 380.8KB/s 51.4s<8.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 92% ━━━━━━━━━━━─ 39.1/42.2MB 379.0KB/s 51.7s<8.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 92% ━━━━━━━━━━━─ 39.3/42.2MB 469.5KB/s 51.9s<6.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 93% ━━━━━━━━━━━─ 39.3/42.2MB 468.0KB/s 52.1s<6.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 93% ━━━━━━━━━━━─ 39.4/42.2MB 439.1KB/s 52.2s<6.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 93% ━━━━━━━━━━━─ 39.4/42.2MB 428.1KB/s 52.3s<6.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 93% ━━━━━━━━━━━─ 39.5/42.2MB 436.2KB/s 52.5s<6.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 93% ━━━━━━━━━━━─ 39.5/42.2MB 414.4KB/s 52.6s<6.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 93% ━━━━━━━━━━━─ 39.6/42.2MB 431.7KB/s 52.8s<6.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 93% ━━━━━━━━━━━─ 39.7/42.2MB 411.5KB/s 52.9s<6.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 94% ━━━━━━━━━━━─ 39.8/42.2MB 422.0KB/s 53.1s<5.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 94% ━━━━━━━━━━━─ 39.8/42.2MB 415.2KB/s 53.2s<5.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 94% ━━━━━━━━━━━─ 39.8/42.2MB 403.6KB/s 53.3s<6.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 94% ━━━━━━━━━━━─ 39.9/42.2MB 418.1KB/s 53.5s<5.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 94% ━━━━━━━━━━━─ 40.0/42.2MB 400.8KB/s 53.7s<5.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 94% ━━━━━━━━━━━─ 40.1/42.2MB 430.2KB/s 53.8s<5.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 94% ━━━━━━━━━━━─ 40.1/42.2MB 426.9KB/s 53.9s<5.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 95% ━━━━━━━━━━━─ 40.2/42.2MB 377.1KB/s 54.3s<5.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 95% ━━━━━━━━━━━─ 40.3/42.2MB 505.1KB/s 54.4s<3.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 95% ━━━━━━━━━━━─ 40.3/42.2MB 469.6KB/s 54.5s<4.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 95% ━━━━━━━━━━━─ 40.4/42.2MB 440.3KB/s 54.7s<4.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 95% ━━━━━━━━━━━╸ 40.5/42.2MB 463.2KB/s 54.8s<3.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 96% ━━━━━━━━━━━╸ 40.6/42.2MB 1.1MB/s 54.9s<1.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 96% ━━━━━━━━━━━╸ 40.8/42.2MB 1.3MB/s 55.1s<1.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 96% ━━━━━━━━━━━╸ 40.9/42.2MB 1.4MB/s 55.2s<0.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 97% ━━━━━━━━━━━╸ 41.1/42.2MB 1.5MB/s 55.3s<0.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 97% ━━━━━━━━━━━╸ 41.3/42.2MB 1.7MB/s 55.4s<0.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 98% ━━━━━━━━━━━╸ 41.5/42.2MB 1.4MB/s 55.5s<0.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 98% ━━━━━━━━━━━╸ 41.7/42.2MB 1.8MB/s 55.6s<0.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 99% ━━━━━━━━━━━╸ 41.9/42.2MB 1.8MB/s 55.8s<0.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 99% ━━━━━━━━━━━╸ 42.0/42.2MB 522.4KB/s 56.0s<0.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 99% ━━━━━━━━━━━╸ 42.1/42.2MB 520.3KB/s 56.1s<0.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 99% ━━━━━━━━━━━╸ 42.2/42.2MB 489.9KB/s 56.2s<0.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt to '/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt': 100% ━━━━━━━━━━━━ 42.2MB 767.5KB/s 56.3s
[*] Fine-tuning YOLO26m on leaf detection...
New https://pypi.org/project/ultralytics/8.4.90 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.4.75 🚀 Python-3.14.6 torch-2.12.1+cu130 CUDA:0 (NVIDIA GeForce RTX 5060 Laptop GPU, 8151MiB)
[34m[1mengine/trainer: [0magnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/dataset/yolo_dataset/leaf_data.yaml, degrees=0.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=5, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=224, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26m.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=yolo26_leaf, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26_train, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26_train/yolo26_leaf, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=1
from  n    params  module                                       arguments
0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]
1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]
3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]
5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]
6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]
7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]
8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]
9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5, 3, True]
10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]
11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]
14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]
17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]
19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]
20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]
21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]
22                  -1  1   1974784  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True, 0.5, True]
23        [16, 19, 22]  1   2800158  ultralytics.nn.modules.head.Detect           [1, 1, True, [256, 512, 512]]
YOLO26m summary: 280 layers, 21,774,430 parameters, 21,774,430 gradients, 74.7 GFLOPs
Transferred 756/768 items from pretrained weights
[34m[1mAMP: [0mrunning Automatic Mixed Precision (AMP) checks...
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 0% ──────────── 48.0KB/5.3MB 131.9KB/s 0.1s<40.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 1% ──────────── 96.0KB/5.3MB 214.3KB/s 0.2s<24.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 2% ──────────── 144.0KB/5.3MB 268.9KB/s 0.3s<19.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 3% ──────────── 192.0KB/5.3MB 312.3KB/s 0.5s<16.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 4% ╸─────────── 240.0KB/5.3MB 342.6KB/s 0.6s<15.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 5% ╸─────────── 288.0KB/5.3MB 358.7KB/s 0.7s<14.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 6% ╸─────────── 336.0KB/5.3MB 382.2KB/s 0.8s<13.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 7% ╸─────────── 384.0KB/5.3MB 394.3KB/s 0.9s<12.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 7% ╸─────────── 432.0KB/5.3MB 399.0KB/s 1.0s<12.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 8% ━─────────── 480.0KB/5.3MB 401.9KB/s 1.2s<12.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 9% ━─────────── 528.0KB/5.3MB 417.2KB/s 1.3s<11.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 10% ━─────────── 576.0KB/5.3MB 399.8KB/s 1.4s<12.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 11% ━─────────── 624.0KB/5.3MB 409.1KB/s 1.5s<11.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 12% ━─────────── 672.0KB/5.3MB 408.5KB/s 1.6s<11.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 13% ━╸────────── 720.0KB/5.3MB 412.2KB/s 1.7s<11.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 14% ━╸────────── 768.0KB/5.3MB 412.4KB/s 1.9s<11.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 15% ━╸────────── 816.0KB/5.3MB 419.2KB/s 2.0s<11.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 15% ━╸────────── 864.0KB/5.3MB 423.2KB/s 2.1s<10.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 16% ━━────────── 912.0KB/5.3MB 430.9KB/s 2.2s<10.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 17% ━━────────── 960.0KB/5.3MB 427.9KB/s 2.3s<10.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 18% ━━────────── 1008.0KB/5.3MB 412.5KB/s 2.4s<10.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 19% ━━────────── 1.0/5.3MB 421.9KB/s 2.5s<10.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 20% ━━────────── 1.1/5.3MB 424.1KB/s 2.6s<10.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 21% ━━╸───────── 1.1/5.3MB 422.5KB/s 2.8s<10.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 22% ━━╸───────── 1.2/5.3MB 437.0KB/s 2.9s<9.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 23% ━━╸───────── 1.2/5.3MB 426.0KB/s 3.0s<9.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 23% ━━╸───────── 1.3/5.3MB 441.0KB/s 3.1s<9.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 24% ━━╸───────── 1.3/5.3MB 437.3KB/s 3.2s<9.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 25% ━━━───────── 1.4/5.3MB 439.4KB/s 3.3s<9.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 26% ━━━───────── 1.4/5.3MB 434.4KB/s 3.4s<9.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 27% ━━━───────── 1.5/5.3MB 440.0KB/s 3.5s<8.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 27% ━━━───────── 1.5/5.3MB 320.8KB/s 3.9s<12.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 31% ━━━╸──────── 1.6/5.3MB 1.3MB/s 4.0s<2.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 31% ━━━╸──────── 1.7/5.3MB 341.5KB/s 4.2s<10.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 35% ━━━━──────── 1.9/5.3MB 1.2MB/s 4.3s<2.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 42% ━━━━━─────── 2.2/5.3MB 3.7MB/s 4.4s<0.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 46% ━━━━━╸────── 2.5/5.3MB 2.4MB/s 4.5s<1.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 52% ━━━━━━────── 2.8/5.3MB 2.7MB/s 4.6s<0.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 56% ━━━━━━╸───── 3.0/5.3MB 2.2MB/s 4.7s<1.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 60% ━━━━━━━───── 3.2/5.3MB 2.0MB/s 4.8s<1.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 64% ━━━━━━━╸──── 3.4/5.3MB 1.8MB/s 4.9s<1.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 67% ━━━━━━━━──── 3.5/5.3MB 1.4MB/s 5.0s<1.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 69% ━━━━━━━━──── 3.7/5.3MB 1020.4KB/s 5.2s<1.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 70% ━━━━━━━━──── 3.7/5.3MB 380.4KB/s 5.3s<4.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 71% ━━━━━━━━╸─── 3.8/5.3MB 407.3KB/s 5.4s<3.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 72% ━━━━━━━━╸─── 3.8/5.3MB 421.5KB/s 5.5s<3.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 72% ━━━━━━━━╸─── 3.8/5.3MB 329.1KB/s 5.6s<4.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 74% ━━━━━━━━╸─── 3.9/5.3MB 448.3KB/s 5.7s<3.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 74% ━━━━━━━━╸─── 3.9/5.3MB 356.6KB/s 5.9s<3.9s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 75% ━━━━━━━━━─── 4.0/5.3MB 441.3KB/s 6.0s<3.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 76% ━━━━━━━━━─── 4.1/5.3MB 446.1KB/s 6.1s<2.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 78% ━━━━━━━━━─── 4.1/5.3MB 445.6KB/s 6.2s<2.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 79% ━━━━━━━━━╸── 4.2/5.3MB 459.6KB/s 6.4s<2.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 80% ━━━━━━━━━╸── 4.2/5.3MB 454.3KB/s 6.5s<2.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 80% ━━━━━━━━━╸── 4.3/5.3MB 449.3KB/s 6.6s<2.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 81% ━━━━━━━━━╸── 4.3/5.3MB 450.6KB/s 6.7s<2.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 82% ━━━━━━━━━╸── 4.4/5.3MB 453.7KB/s 6.8s<2.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 83% ━━━━━━━━━━── 4.4/5.3MB 452.9KB/s 6.9s<2.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 84% ━━━━━━━━━━── 4.5/5.3MB 458.2KB/s 7.0s<1.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 85% ━━━━━━━━━━── 4.5/5.3MB 459.0KB/s 7.1s<1.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 86% ━━━━━━━━━━── 4.6/5.3MB 440.9KB/s 7.2s<1.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 87% ━━━━━━━━━━── 4.6/5.3MB 444.4KB/s 7.3s<1.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 88% ━━━━━━━━━━╸─ 4.7/5.3MB 445.1KB/s 7.4s<1.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 88% ━━━━━━━━━━╸─ 4.7/5.3MB 433.6KB/s 7.5s<1.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 89% ━━━━━━━━━━╸─ 4.8/5.3MB 432.1KB/s 7.7s<1.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 90% ━━━━━━━━━━╸─ 4.8/5.3MB 428.9KB/s 7.8s<1.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 91% ━━━━━━━━━━╸─ 4.8/5.3MB 428.4KB/s 7.9s<1.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 92% ━━━━━━━━━━━─ 4.9/5.3MB 423.8KB/s 8.0s<1.0s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 93% ━━━━━━━━━━━─ 4.9/5.3MB 421.8KB/s 8.1s<0.8s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 94% ━━━━━━━━━━━─ 5.0/5.3MB 419.1KB/s 8.2s<0.7s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 95% ━━━━━━━━━━━─ 5.0/5.3MB 413.6KB/s 8.4s<0.6s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 96% ━━━━━━━━━━━╸ 5.1/5.3MB 423.5KB/s 8.5s<0.5s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 96% ━━━━━━━━━━━╸ 5.1/5.3MB 418.6KB/s 8.6s<0.4s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 97% ━━━━━━━━━━━╸ 5.2/5.3MB 420.5KB/s 8.7s<0.3s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 98% ━━━━━━━━━━━╸ 5.2/5.3MB 419.8KB/s 8.8s<0.2s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 99% ━━━━━━━━━━━╸ 5.3/5.3MB 419.3KB/s 8.9s<0.1s
[KDownloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 605.9KB/s 8.9s
[34m[1mAMP: [0mchecks passed ✅
[34m[1mtrain: [0mFast image access ✅ (ping: 0.0±0.0 ms, read: 2710.2±3944.6 MB/s, size: 326.9 KB)
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 72 images, 0 backgrounds, 0 corrupt: 4% ──────────── 72/1796 211.9it/s 0.1s<8.1s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 177 images, 0 backgrounds, 0 corrupt: 9% ━─────────── 177/1796 456.2it/s 0.2s<3.5s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 282 images, 0 backgrounds, 0 corrupt: 15% ━╸────────── 282/1796 633.2it/s 0.3s<2.4s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 388 images, 0 backgrounds, 0 corrupt: 21% ━━╸───────── 388/1796 746.2it/s 0.4s<1.9s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 493 images, 0 backgrounds, 0 corrupt: 27% ━━━───────── 493/1796 835.0it/s 0.5s<1.6s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 602 images, 0 backgrounds, 0 corrupt: 33% ━━━━──────── 602/1796 905.4it/s 0.6s<1.3s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 704 images, 0 backgrounds, 0 corrupt: 39% ━━━━╸─────── 704/1796 928.8it/s 0.7s<1.2s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 800 images, 0 backgrounds, 0 corrupt: 44% ━━━━━─────── 800/1796 932.2it/s 0.8s<1.1s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 901 images, 0 backgrounds, 0 corrupt: 50% ━━━━━━────── 901/1796 954.5it/s 0.9s<0.9s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 1004 images, 0 backgrounds, 0 corrupt: 55% ━━━━━━╸───── 1004/1796 970.1it/s 1.0s<0.8s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 1105 images, 0 backgrounds, 0 corrupt: 61% ━━━━━━━───── 1105/1796 982.0it/s 1.1s<0.7s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 1212 images, 0 backgrounds, 0 corrupt: 67% ━━━━━━━━──── 1212/1796 1.0Kit/s 1.2s<0.6s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 1314 images, 0 backgrounds, 0 corrupt: 73% ━━━━━━━━╸─── 1314/1796 998.1it/s 1.3s<0.5s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 1416 images, 0 backgrounds, 0 corrupt: 78% ━━━━━━━━━─── 1416/1796 1.0Kit/s 1.4s<0.4s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 1521 images, 0 backgrounds, 0 corrupt: 84% ━━━━━━━━━━── 1521/1796 1.0Kit/s 1.5s<0.3s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 1625 images, 0 backgrounds, 0 corrupt: 90% ━━━━━━━━━━╸─ 1625/1796 1.0Kit/s 1.6s<0.2s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 1731 images, 0 backgrounds, 0 corrupt: 96% ━━━━━━━━━━━╸ 1731/1796 1.0Kit/s 1.7s<0.1s
[K[34m[1mtrain: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train... 1796 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1796/1796 1.0Kit/s 1.8s
[34m[1mtrain: [0mNew cache created: /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/train.cache
[34m[1mval: [0mFast image access ✅ (ping: 0.0±0.0 ms, read: 854.2±707.5 MB/s, size: 15.0 KB)
[K[34m[1mval: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/val... 73 images, 0 backgrounds, 0 corrupt: 35% ━━━━──────── 73/204 217.7it/s 0.1s<0.6s
[K[34m[1mval: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/val... 135 images, 0 backgrounds, 0 corrupt: 66% ━━━━━━━╸──── 135/204 330.5it/s 0.2s<0.2s
[K[34m[1mval: [0mScanning /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/val... 204 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 204/204 670.6it/s 0.3s
[34m[1mval: [0mNew cache created: /home/swapnil/leaf_disease_dataset/dataset/yolo_dataset/labels/val.cache
[34m[1moptimizer:[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
[34m[1moptimizer:[0m AdamW(lr=0.002, momentum=0.9) with parameter groups 124 weight(decay=0.0), 136 weight(decay=0.0005), 136 bias(decay=0.0)
Plotting labels to /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26_train/yolo26_leaf/labels.jpg...
Image sizes 224 train, 224 val
Using 8 dataloader workers
Logging results to [1m/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26_train/yolo26_leaf[0m
Starting training for 5 epochs...
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
[K        1/5      2.41G      1.053      2.406    0.03031         92        224: 0% ──────────── 0/57  3.8s
[K        1/5      2.41G      1.114      2.388    0.03495         91        224: 1% ──────────── 1/57 1.5it/s 4.0s<37.7s
[K        1/5      2.41G      1.079      2.438     0.0332         91        224: 3% ──────────── 2/57 2.8it/s 4.2s<19.4s
[K        1/5      2.41G      1.071      2.426    0.03332         96        224: 5% ╸─────────── 3/57 3.7it/s 4.3s<14.6s
[K        1/5      2.41G      1.086      2.426    0.03397         92        224: 7% ╸─────────── 4/57 4.4it/s 4.5s<11.9s
[K        1/5      2.41G      1.093       2.45    0.03398         87        224: 8% ━─────────── 5/57 4.9it/s 4.7s<10.5s
[K        1/5      2.41G      1.093      2.461    0.03399         85        224: 10% ━─────────── 6/57 5.3it/s 4.8s<9.7s
[K        1/5      2.45G      1.092      2.468    0.03424         85        224: 12% ━─────────── 7/57 4.6it/s 5.1s<10.8s
[K        1/5      2.47G      1.099      2.451    0.03427         91        224: 14% ━╸────────── 8/57 4.9it/s 5.3s<10.0s
[K        1/5      2.47G      1.086      2.429    0.03385         99        224: 15% ━╸────────── 9/57 5.2it/s 5.5s<9.2s
[K        1/5      2.47G       1.07      2.406    0.03321         94        224: 17% ━━────────── 10/57 5.4it/s 5.7s<8.8s
[K        1/5      2.47G       1.05      2.381     0.0326         92        224: 19% ━━────────── 11/57 5.5it/s 5.8s<8.3s
[K        1/5      2.47G      1.031      2.345    0.03182         99        224: 21% ━━╸───────── 12/57 5.7it/s 6.0s<7.9s
[K        1/5      2.47G      1.013      2.307    0.03112         91        224: 22% ━━╸───────── 13/57 5.8it/s 6.2s<7.6s
[K        1/5      2.47G     0.9942      2.264     0.0305         89        224: 24% ━━╸───────── 14/57 5.8it/s 6.3s<7.4s
[K        1/5      2.47G     0.9779      2.222    0.02986         92        224: 26% ━━━───────── 15/57 5.9it/s 6.5s<7.2s
[K        1/5      2.47G      0.962      2.175    0.02934         93        224: 28% ━━━───────── 16/57 5.9it/s 6.7s<6.9s
[K        1/5      2.47G     0.9478      2.139    0.02888         86        224: 29% ━━━╸──────── 17/57 5.9it/s 6.8s<6.8s
[K        1/5      2.47G     0.9432      2.113    0.02853         86        224: 31% ━━━╸──────── 18/57 5.8it/s 7.0s<6.7s
[K        1/5      2.47G     0.9296      2.073      0.028        103        224: 33% ━━━━──────── 19/57 5.9it/s 7.2s<6.5s
[K        1/5      2.47G      0.913      2.027    0.02753         86        224: 35% ━━━━──────── 20/57 5.9it/s 7.4s<6.3s
[K        1/5      2.47G     0.9063      1.999    0.02722         92        224: 36% ━━━━──────── 21/57 5.9it/s 7.5s<6.1s
[K        1/5      2.47G     0.8991      1.967    0.02694         90        224: 38% ━━━━╸─────── 22/57 5.8it/s 7.7s<6.0s
[K        1/5      2.47G      0.891      1.932    0.02663         98        224: 40% ━━━━╸─────── 23/57 5.9it/s 7.9s<5.8s
[K        1/5      2.47G     0.8834      1.904     0.0264         84        224: 42% ━━━━━─────── 24/57 5.9it/s 8.0s<5.5s
[K        1/5      2.47G     0.8733      1.872    0.02605         96        224: 43% ━━━━━─────── 25/57 6.0it/s 8.2s<5.4s
[K        1/5      2.47G     0.8665      1.841    0.02581         93        224: 45% ━━━━━─────── 26/57 5.9it/s 8.4s<5.2s
[K        1/5      2.47G     0.8595      1.809    0.02556         91        224: 47% ━━━━━╸────── 27/57 6.0it/s 8.5s<5.0s
[K        1/5      2.47G     0.8523      1.786     0.0253         87        224: 49% ━━━━━╸────── 28/57 6.0it/s 8.7s<4.9s
[K        1/5      2.47G     0.8462      1.755     0.0251         93        224: 50% ━━━━━━────── 29/57 5.8it/s 8.9s<4.8s
[K        1/5      2.47G     0.8418      1.729     0.0249         95        224: 52% ━━━━━━────── 30/57 5.9it/s 9.0s<4.6s
[K        1/5      2.47G     0.8375      1.704    0.02476         93        224: 54% ━━━━━━╸───── 31/57 5.9it/s 9.2s<4.4s
[K        1/5      2.47G     0.8313      1.679    0.02452        100        224: 56% ━━━━━━╸───── 32/57 5.9it/s 9.4s<4.2s
[K        1/5      2.47G     0.8248      1.655    0.02429         87        224: 57% ━━━━━━╸───── 33/57 5.9it/s 9.6s<4.0s
[K        1/5      2.47G     0.8219      1.637    0.02413         93        224: 59% ━━━━━━━───── 34/57 6.0it/s 9.7s<3.8s
[K        1/5      2.47G     0.8184      1.616    0.02401         91        224: 61% ━━━━━━━───── 35/57 6.0it/s 9.9s<3.7s
[K        1/5      2.47G     0.8185      1.596    0.02399        101        224: 63% ━━━━━━━╸──── 36/57 6.0it/s 10.0s<3.5s
[K        1/5      2.47G     0.8187      1.576    0.02398         95        224: 64% ━━━━━━━╸──── 37/57 6.0it/s 10.2s<3.3s
[K        1/5      2.47G     0.8191      1.558    0.02402         88        224: 66% ━━━━━━━━──── 38/57 6.0it/s 10.4s<3.2s
[K        1/5      2.47G     0.8188       1.54    0.02398         93        224: 68% ━━━━━━━━──── 39/57 6.0it/s 10.5s<3.0s
[K        1/5      2.47G     0.8185      1.523    0.02395        102        224: 70% ━━━━━━━━──── 40/57 6.0it/s 10.7s<2.8s
[K        1/5      2.47G     0.8168      1.507    0.02386        104        224: 71% ━━━━━━━━╸─── 41/57 6.0it/s 10.9s<2.7s
[K        1/5      2.47G     0.8187      1.492    0.02385         95        224: 73% ━━━━━━━━╸─── 42/57 5.9it/s 11.1s<2.5s
[K        1/5      2.47G     0.8184      1.475    0.02378         91        224: 75% ━━━━━━━━━─── 43/57 6.0it/s 11.2s<2.3s
[K        1/5      2.47G     0.8177      1.458    0.02373         88        224: 77% ━━━━━━━━━─── 44/57 6.0it/s 11.4s<2.2s
[K        1/5      2.47G     0.8185      1.446    0.02371         91        224: 78% ━━━━━━━━━─── 45/57 6.0it/s 11.5s<2.0s
[K        1/5      2.47G     0.8196      1.432    0.02376         80        224: 80% ━━━━━━━━━╸── 46/57 6.0it/s 11.7s<1.8s
[K        1/5      2.47G     0.8218      1.417    0.02381         88        224: 82% ━━━━━━━━━╸── 47/57 6.0it/s 11.9s<1.7s
[K        1/5      2.47G     0.8208      1.402    0.02374        101        224: 84% ━━━━━━━━━━── 48/57 6.0it/s 12.0s<1.5s
[K        1/5      2.47G     0.8185      1.386    0.02366         92        224: 85% ━━━━━━━━━━── 49/57 6.0it/s 12.2s<1.3s
[K        1/5      2.47G     0.8189      1.375    0.02363         94        224: 87% ━━━━━━━━━━╸─ 50/57 6.0it/s 12.4s<1.2s
[K        1/5      2.47G     0.8201      1.366    0.02365         92        224: 89% ━━━━━━━━━━╸─ 51/57 6.0it/s 12.5s<1.0s
[K        1/5      2.47G     0.8217      1.354    0.02371         91        224: 91% ━━━━━━━━━━╸─ 52/57 6.0it/s 12.7s<0.8s
[K        1/5      2.47G     0.8191      1.341    0.02361         90        224: 92% ━━━━━━━━━━━─ 53/57 6.0it/s 12.9s<0.7s
[K        1/5      2.47G     0.8215      1.331    0.02366        107        224: 94% ━━━━━━━━━━━─ 54/57 6.0it/s 13.1s<0.5s
[K        1/5      2.47G     0.8208      1.318    0.02364        100        224: 96% ━━━━━━━━━━━╸ 55/57 6.0it/s 13.2s<0.3s
[K        1/5      2.51G     0.8268      1.312    0.02369         12        224: 98% ━━━━━━━━━━━╸ 56/57 4.4it/s 14.9s<0.2s
[K        1/5      2.51G     0.8268      1.312    0.02369         12        224: 100% ━━━━━━━━━━━━ 57/57 3.8it/s 14.9s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 25% ━━━───────── 1/4 2.3s/it 0.7s<6.8s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 50% ━━━━━━────── 2/4 1.9it/s 0.9s<1.0s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 75% ━━━━━━━━━─── 3/4 3.0it/s 1.0s<0.3s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 2.6it/s 1.6s
all        204        204     0.0367     0.0294    0.00303   0.000496
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
[K        2/5       3.2G     0.8158     0.6241    0.02334         87        224: 0% ──────────── 0/57  0.2s
[K        2/5       3.2G     0.8496     0.6029    0.02369         93        224: 1% ──────────── 1/57 1.7it/s 0.4s<32.7s
[K        2/5       3.2G     0.8065     0.6256    0.02193        100        224: 3% ──────────── 2/57 2.9it/s 0.5s<19.2s
[K        2/5       3.2G     0.8281     0.6127    0.02242         93        224: 5% ╸─────────── 3/57 3.6it/s 0.7s<14.9s
[K        2/5       3.2G     0.8263     0.6186    0.02223         98        224: 7% ╸─────────── 4/57 4.2it/s 0.9s<12.5s
[K        2/5       3.2G     0.8344     0.6331    0.02242        105        224: 8% ━─────────── 5/57 4.7it/s 1.1s<11.2s
[K        2/5       3.2G     0.8428      0.631    0.02273        103        224: 10% ━─────────── 6/57 5.0it/s 1.2s<10.3s
[K        2/5       3.2G     0.8496     0.6301    0.02304         97        224: 12% ━─────────── 7/57 5.1it/s 1.4s<9.7s
[K        2/5       3.2G     0.8494     0.6326    0.02318         96        224: 14% ━╸────────── 8/57 5.3it/s 1.6s<9.3s
[K        2/5       3.2G     0.8559     0.6373    0.02332         96        224: 15% ━╸────────── 9/57 5.4it/s 1.8s<8.9s
[K        2/5       3.2G      0.855     0.6424    0.02325         93        224: 17% ━━────────── 10/57 5.4it/s 2.0s<8.6s
[K        2/5       3.2G     0.8602     0.6527    0.02357         87        224: 19% ━━────────── 11/57 5.5it/s 2.1s<8.4s
[K        2/5       3.2G     0.8552     0.6582    0.02344         92        224: 21% ━━╸───────── 12/57 5.6it/s 2.3s<8.1s
[K        2/5       3.2G      0.848     0.6504    0.02313        101        224: 22% ━━╸───────── 13/57 5.6it/s 2.5s<7.9s
[K        2/5       3.2G     0.8443     0.6514    0.02304         91        224: 24% ━━╸───────── 14/57 5.5it/s 2.7s<7.8s
[K        2/5       3.2G      0.844      0.658    0.02316         87        224: 26% ━━━───────── 15/57 5.5it/s 2.9s<7.6s
[K        2/5       3.2G     0.8346     0.6564    0.02297         90        224: 28% ━━━───────── 16/57 5.5it/s 3.0s<7.4s
[K        2/5       3.2G     0.8375     0.6508    0.02303         96        224: 29% ━━━╸──────── 17/57 5.6it/s 3.2s<7.2s
[K        2/5       3.2G     0.8421     0.6502     0.0232        100        224: 31% ━━━╸──────── 18/57 5.6it/s 3.4s<7.0s
[K        2/5       3.2G     0.8401     0.6531    0.02314         89        224: 33% ━━━━──────── 19/57 5.6it/s 3.6s<6.8s
[K        2/5       3.2G     0.8355     0.6463    0.02295         99        224: 35% ━━━━──────── 20/57 5.6it/s 3.8s<6.6s
[K        2/5       3.2G     0.8332     0.6457    0.02296         84        224: 36% ━━━━──────── 21/57 5.6it/s 3.9s<6.5s
[K        2/5       3.2G     0.8311     0.6372    0.02288        104        224: 38% ━━━━╸─────── 22/57 5.5it/s 4.1s<6.3s
[K        2/5       3.2G     0.8312     0.6357    0.02289         90        224: 40% ━━━━╸─────── 23/57 5.6it/s 4.3s<6.1s
[K        2/5       3.2G     0.8325     0.6342    0.02296         96        224: 42% ━━━━━─────── 24/57 5.6it/s 4.5s<5.9s
[K        2/5       3.2G     0.8338     0.6325    0.02301         92        224: 43% ━━━━━─────── 25/57 5.6it/s 4.7s<5.7s
[K        2/5       3.2G     0.8327     0.6368    0.02301         97        224: 45% ━━━━━─────── 26/57 5.5it/s 4.8s<5.7s
[K        2/5       3.2G     0.8279     0.6397    0.02288         96        224: 47% ━━━━━╸────── 27/57 5.5it/s 5.0s<5.5s
[K        2/5       3.2G     0.8248     0.6385    0.02276         93        224: 49% ━━━━━╸────── 28/57 5.5it/s 5.2s<5.3s
[K        2/5       3.2G     0.8235     0.6358    0.02272         92        224: 50% ━━━━━━────── 29/57 5.9it/s 5.4s<4.7s
[K        2/5       3.2G      0.827     0.6379    0.02286         87        224: 52% ━━━━━━────── 30/57 5.8it/s 5.5s<4.7s
[K        2/5       3.2G     0.8271     0.6391    0.02285        101        224: 54% ━━━━━━╸───── 31/57 6.1it/s 5.7s<4.2s
[K        2/5       3.2G     0.8274     0.6367    0.02293         86        224: 56% ━━━━━━╸───── 32/57 5.9it/s 5.9s<4.2s
[K        2/5       3.2G     0.8249     0.6333    0.02286         88        224: 57% ━━━━━━╸───── 33/57 6.2it/s 6.0s<3.9s
[K        2/5       3.2G      0.822      0.629    0.02279         97        224: 59% ━━━━━━━───── 34/57 6.0it/s 6.2s<3.8s
[K        2/5       3.2G     0.8185     0.6256    0.02269         93        224: 61% ━━━━━━━───── 35/57 6.3it/s 6.3s<3.5s
[K        2/5       3.2G     0.8189     0.6233    0.02272         94        224: 63% ━━━━━━━╸──── 36/57 6.1it/s 6.5s<3.5s
[K        2/5       3.2G     0.8179      0.621     0.0227         92        224: 64% ━━━━━━━╸──── 37/57 6.3it/s 6.7s<3.2s
[K        2/5       3.2G     0.8201     0.6209    0.02276         99        224: 66% ━━━━━━━━──── 38/57 6.0it/s 6.8s<3.1s
[K        2/5       3.2G     0.8188     0.6192    0.02274         92        224: 68% ━━━━━━━━──── 39/57 6.2it/s 7.0s<2.9s
[K        2/5       3.2G     0.8191     0.6178    0.02277         78        224: 70% ━━━━━━━━──── 40/57 6.0it/s 7.2s<2.8s
[K        2/5       3.2G     0.8184     0.6178    0.02273         97        224: 71% ━━━━━━━━╸─── 41/57 6.4it/s 7.3s<2.5s
[K        2/5       3.2G     0.8182     0.6148    0.02269        102        224: 73% ━━━━━━━━╸─── 42/57 6.1it/s 7.5s<2.5s
[K        2/5       3.2G     0.8144      0.611    0.02257         99        224: 75% ━━━━━━━━━─── 43/57 6.5it/s 7.6s<2.2s
[K        2/5       3.2G     0.8133      0.609    0.02253         93        224: 77% ━━━━━━━━━─── 44/57 6.2it/s 7.8s<2.1s
[K        2/5       3.2G     0.8126     0.6079    0.02248         94        224: 78% ━━━━━━━━━─── 45/57 6.4it/s 8.0s<1.9s
[K        2/5       3.2G      0.812     0.6063    0.02253         73        224: 80% ━━━━━━━━━╸── 46/57 6.1it/s 8.1s<1.8s
[K        2/5       3.2G     0.8115     0.6041    0.02251         94        224: 82% ━━━━━━━━━╸── 47/57 6.5it/s 8.3s<1.5s
[K        2/5       3.2G     0.8108     0.6034    0.02249        100        224: 84% ━━━━━━━━━━── 48/57 6.2it/s 8.5s<1.5s
[K        2/5       3.2G     0.8106     0.6033    0.02245         93        224: 85% ━━━━━━━━━━── 49/57 6.5it/s 8.6s<1.2s
[K        2/5       3.2G     0.8086     0.6014    0.02237         89        224: 87% ━━━━━━━━━━╸─ 50/57 6.2it/s 8.8s<1.1s
[K        2/5       3.2G     0.8078     0.6008     0.0223        108        224: 89% ━━━━━━━━━━╸─ 51/57 6.4it/s 8.9s<0.9s
[K        2/5       3.2G      0.808     0.6008    0.02227         94        224: 91% ━━━━━━━━━━╸─ 52/57 6.2it/s 9.1s<0.8s
[K        2/5       3.2G     0.8091        0.6    0.02227         88        224: 92% ━━━━━━━━━━━─ 53/57 6.5it/s 9.2s<0.6s
[K        2/5       3.2G     0.8119     0.6013    0.02231         89        224: 94% ━━━━━━━━━━━─ 54/57 6.2it/s 9.4s<0.5s
[K        2/5       3.2G     0.8132     0.5989    0.02239         86        224: 96% ━━━━━━━━━━━╸ 55/57 6.5it/s 9.6s<0.3s
[K        2/5      3.22G     0.8204     0.6042    0.02263          9        224: 98% ━━━━━━━━━━━╸ 56/57 7.1it/s 9.7s<0.1s
[K        2/5      3.22G     0.8204     0.6042    0.02263          9        224: 100% ━━━━━━━━━━━━ 57/57 5.9it/s 9.7s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 25% ━━━───────── 1/4 1.5it/s 0.2s<2.0s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 50% ━━━━━━────── 2/4 2.6it/s 0.4s<0.8s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 75% ━━━━━━━━━─── 3/4 3.4it/s 0.6s<0.3s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 6.3it/s 0.6s
all        204        204      0.921      0.922      0.965      0.493
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
[K        3/5      3.22G     0.8429     0.4876    0.02447         89        224: 0% ──────────── 0/57  0.1s
[K        3/5      3.22G     0.8149     0.4878    0.02305         85        224: 1% ──────────── 1/57 1.7it/s 0.3s<33.7s
[K        3/5      3.22G     0.7637     0.5214    0.02216         89        224: 3% ──────────── 2/57 3.3it/s 0.5s<16.7s
[K        3/5      3.22G     0.7548      0.515     0.0217         84        224: 5% ╸─────────── 3/57 3.9it/s 0.6s<13.8s
[K        3/5      3.22G     0.7482     0.5091    0.02121         96        224: 7% ╸─────────── 4/57 4.9it/s 0.8s<10.8s
[K        3/5      3.22G     0.7549     0.5142    0.02121         94        224: 8% ━─────────── 5/57 5.0it/s 1.0s<10.3s
[K        3/5      3.22G     0.7384     0.5083    0.02072         95        224: 10% ━─────────── 6/57 5.7it/s 1.1s<9.0s
[K        3/5      3.22G     0.7408     0.5152    0.02087         86        224: 12% ━─────────── 7/57 5.6it/s 1.3s<9.0s
[K        3/5      3.22G     0.7422     0.5105    0.02087         95        224: 14% ━╸────────── 8/57 6.1it/s 1.4s<8.1s
[K        3/5      3.22G      0.749     0.5124     0.0211         88        224: 15% ━╸────────── 9/57 5.9it/s 1.6s<8.1s
[K        3/5      3.22G     0.7535     0.5109    0.02118        103        224: 17% ━━────────── 10/57 6.3it/s 1.8s<7.5s
[K        3/5      3.22G     0.7529      0.509    0.02117         94        224: 19% ━━────────── 11/57 6.0it/s 1.9s<7.7s
[K        3/5      3.22G      0.759     0.5089    0.02149         90        224: 21% ━━╸───────── 12/57 6.3it/s 2.1s<7.1s
[K        3/5      3.22G     0.7595     0.5115    0.02156         94        224: 22% ━━╸───────── 13/57 6.1it/s 2.3s<7.3s
[K        3/5      3.22G     0.7509     0.5073    0.02125         97        224: 24% ━━╸───────── 14/57 6.4it/s 2.4s<6.7s
[K        3/5      3.22G     0.7544     0.5068    0.02135         94        224: 26% ━━━───────── 15/57 6.1it/s 2.6s<6.8s
[K        3/5      3.22G     0.7532     0.5132    0.02131         91        224: 28% ━━━───────── 16/57 6.4it/s 2.7s<6.4s
[K        3/5      3.22G     0.7492     0.5132    0.02125         96        224: 29% ━━━╸──────── 17/57 6.2it/s 2.9s<6.5s
[K        3/5      3.22G      0.752     0.5145    0.02128         99        224: 31% ━━━╸──────── 18/57 6.5it/s 3.1s<6.0s
[K        3/5      3.22G     0.7598     0.5132    0.02152         98        224: 33% ━━━━──────── 19/57 6.2it/s 3.2s<6.2s
[K        3/5      3.22G      0.761     0.5108    0.02156         95        224: 35% ━━━━──────── 20/57 6.5it/s 3.4s<5.7s
[K        3/5      3.22G      0.759     0.5124    0.02148        102        224: 36% ━━━━──────── 21/57 6.2it/s 3.6s<5.8s
[K        3/5      3.22G      0.757     0.5077    0.02142         90        224: 38% ━━━━╸─────── 22/57 6.5it/s 3.7s<5.4s
[K        3/5      3.22G     0.7579     0.5096    0.02147         91        224: 40% ━━━━╸─────── 23/57 6.2it/s 3.9s<5.5s
[K        3/5      3.22G     0.7538     0.5078    0.02134         85        224: 42% ━━━━━─────── 24/57 6.5it/s 4.0s<5.0s
[K        3/5      3.22G      0.753     0.5119    0.02122        101        224: 43% ━━━━━─────── 25/57 6.2it/s 4.2s<5.1s
[K        3/5      3.22G     0.7534     0.5111     0.0211        102        224: 45% ━━━━━─────── 26/57 6.5it/s 4.3s<4.7s
[K        3/5      3.22G     0.7544     0.5082     0.0211        100        224: 47% ━━━━━╸────── 27/57 6.2it/s 4.5s<4.8s
[K        3/5      3.22G     0.7557     0.5052     0.0211         94        224: 49% ━━━━━╸────── 28/57 6.5it/s 4.7s<4.4s
[K        3/5      3.22G     0.7577     0.5054    0.02114         91        224: 50% ━━━━━━────── 29/57 6.2it/s 4.8s<4.5s
[K        3/5      3.22G     0.7563     0.5023    0.02106        105        224: 52% ━━━━━━────── 30/57 6.5it/s 5.0s<4.1s
[K        3/5      3.22G     0.7545     0.4996    0.02097         95        224: 54% ━━━━━━╸───── 31/57 6.2it/s 5.2s<4.2s
[K        3/5      3.22G     0.7505     0.4941    0.02085        102        224: 56% ━━━━━━╸───── 32/57 6.5it/s 5.3s<3.8s
[K        3/5      3.22G     0.7506     0.4937     0.0208        107        224: 57% ━━━━━━╸───── 33/57 6.2it/s 5.5s<3.9s
[K        3/5      3.22G     0.7479     0.4922    0.02072         99        224: 59% ━━━━━━━───── 34/57 6.6it/s 5.6s<3.5s
[K        3/5      3.22G     0.7469     0.4911    0.02069        105        224: 61% ━━━━━━━───── 35/57 6.2it/s 5.8s<3.5s
[K        3/5      3.22G     0.7448     0.4909    0.02066         90        224: 63% ━━━━━━━╸──── 36/57 6.4it/s 5.9s<3.3s
[K        3/5      3.22G     0.7464     0.4984    0.02073         73        224: 64% ━━━━━━━╸──── 37/57 6.1it/s 6.1s<3.3s
[K        3/5      3.22G     0.7489     0.4983    0.02083         88        224: 66% ━━━━━━━━──── 38/57 6.5it/s 6.3s<2.9s
[K        3/5      3.22G      0.749     0.4976    0.02084         90        224: 68% ━━━━━━━━──── 39/57 6.2it/s 6.4s<2.9s
[K        3/5      3.22G     0.7465     0.4946    0.02081         90        224: 70% ━━━━━━━━──── 40/57 6.6it/s 6.6s<2.6s
[K        3/5      3.22G     0.7463     0.4951    0.02079         97        224: 71% ━━━━━━━━╸─── 41/57 6.3it/s 6.8s<2.6s
[K        3/5      3.22G     0.7459     0.4959    0.02075         91        224: 73% ━━━━━━━━╸─── 42/57 6.5it/s 6.9s<2.3s
[K        3/5      3.22G     0.7457     0.4977    0.02073         92        224: 75% ━━━━━━━━━─── 43/57 6.2it/s 7.1s<2.3s
[K        3/5      3.22G     0.7428     0.4952    0.02063        101        224: 77% ━━━━━━━━━─── 44/57 6.4it/s 7.2s<2.0s
[K        3/5      3.22G     0.7435     0.4948    0.02061         94        224: 78% ━━━━━━━━━─── 45/57 6.2it/s 7.4s<1.9s
[K        3/5      3.22G      0.742     0.4943    0.02057         95        224: 80% ━━━━━━━━━╸── 46/57 6.5it/s 7.5s<1.7s
[K        3/5      3.22G     0.7416     0.4915    0.02057         85        224: 82% ━━━━━━━━━╸── 47/57 6.2it/s 7.7s<1.6s
[K        3/5      3.22G     0.7413     0.4919    0.02057         88        224: 84% ━━━━━━━━━━── 48/57 6.3it/s 7.9s<1.4s
[K        3/5      3.22G     0.7399     0.4924    0.02053         94        224: 85% ━━━━━━━━━━── 49/57 6.1it/s 8.1s<1.3s
[K        3/5      3.22G     0.7377     0.4909    0.02045        100        224: 87% ━━━━━━━━━━╸─ 50/57 6.5it/s 8.2s<1.1s
[K        3/5      3.22G     0.7366     0.4896    0.02038         90        224: 89% ━━━━━━━━━━╸─ 51/57 6.2it/s 8.4s<1.0s
[K        3/5      3.22G     0.7362     0.4873    0.02033         96        224: 91% ━━━━━━━━━━╸─ 52/57 6.5it/s 8.5s<0.8s
[K        3/5      3.22G      0.735     0.4864    0.02028        105        224: 92% ━━━━━━━━━━━─ 53/57 6.2it/s 8.7s<0.6s
[K        3/5      3.22G     0.7349     0.4851    0.02025         99        224: 94% ━━━━━━━━━━━─ 54/57 6.5it/s 8.8s<0.5s
[K        3/5      3.22G     0.7352     0.4849    0.02026         88        224: 96% ━━━━━━━━━━━╸ 55/57 6.2it/s 9.0s<0.3s
[K        3/5      3.22G     0.7355     0.4841    0.02025         14        224: 100% ━━━━━━━━━━━━ 57/57 6.3it/s 9.1s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 25% ━━━───────── 1/4 1.6it/s 0.2s<1.9s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 50% ━━━━━━────── 2/4 2.7it/s 0.4s<0.7s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 75% ━━━━━━━━━─── 3/4 3.3it/s 0.6s<0.3s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 6.2it/s 0.6s
all        204        204      0.939      0.922      0.972       0.77
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
[K        4/5      3.22G      0.778        0.5    0.02103         87        224: 0% ──────────── 0/57  0.2s
[K        4/5      3.22G     0.6916     0.4638    0.01846        100        224: 1% ──────────── 1/57 2.2it/s 0.3s<25.0s
[K        4/5      3.22G       0.72     0.4802    0.01925         97        224: 3% ──────────── 2/57 3.2it/s 0.5s<16.9s
[K        4/5      3.22G     0.7303     0.4645    0.01935         95        224: 5% ╸─────────── 3/57 4.5it/s 0.6s<12.1s
[K        4/5      3.22G     0.7406     0.4475    0.01944        104        224: 7% ╸─────────── 4/57 4.8it/s 0.8s<11.1s
[K        4/5      3.22G      0.751     0.4559    0.01998         83        224: 8% ━─────────── 5/57 5.3it/s 1.0s<9.8s
[K        4/5      3.22G     0.7529     0.4493    0.02022         95        224: 10% ━─────────── 6/57 5.4it/s 1.1s<9.5s
[K        4/5      3.22G     0.7381     0.4396    0.01995         93        224: 12% ━─────────── 7/57 6.0it/s 1.3s<8.4s
[K        4/5      3.22G     0.7357     0.4427     0.0199         92        224: 14% ━╸────────── 8/57 5.8it/s 1.5s<8.4s
[K        4/5      3.22G     0.7233      0.443    0.01951        102        224: 15% ━╸────────── 9/57 6.2it/s 1.6s<7.7s
[K        4/5      3.22G     0.7291     0.4393    0.01967         85        224: 17% ━━────────── 10/57 6.0it/s 1.8s<7.8s
[K        4/5      3.22G     0.7339     0.4382    0.01977         89        224: 19% ━━────────── 11/57 6.4it/s 1.9s<7.2s
[K        4/5      3.22G     0.7266     0.4314     0.0196         91        224: 21% ━━╸───────── 12/57 6.1it/s 2.1s<7.3s
[K        4/5      3.22G     0.7238     0.4272    0.01948         92        224: 22% ━━╸───────── 13/57 6.5it/s 2.2s<6.8s
[K        4/5      3.22G     0.7261     0.4233    0.01959         91        224: 24% ━━╸───────── 14/57 6.2it/s 2.4s<6.9s
[K        4/5      3.22G     0.7202     0.4202    0.01938         93        224: 26% ━━━───────── 15/57 6.5it/s 2.6s<6.5s
[K        4/5      3.22G     0.7187     0.4263    0.01935         87        224: 28% ━━━───────── 16/57 6.2it/s 2.7s<6.6s
[K        4/5      3.22G     0.7193      0.422    0.01932        103        224: 29% ━━━╸──────── 17/57 6.5it/s 2.9s<6.2s
[K        4/5      3.22G      0.724     0.4223    0.01944        104        224: 31% ━━━╸──────── 18/57 6.2it/s 3.1s<6.3s
[K        4/5      3.22G     0.7305     0.4241    0.01958        107        224: 33% ━━━━──────── 19/57 6.5it/s 3.2s<5.8s
[K        4/5      3.22G     0.7323     0.4213    0.01967         91        224: 35% ━━━━──────── 20/57 6.3it/s 3.4s<5.9s
[K        4/5      3.22G     0.7373     0.4227    0.01982         92        224: 36% ━━━━──────── 21/57 6.6it/s 3.5s<5.5s
[K        4/5      3.22G     0.7384     0.4277    0.01988         95        224: 38% ━━━━╸─────── 22/57 6.1it/s 3.7s<5.8s
[K        4/5      3.22G     0.7359     0.4309    0.01984         83        224: 40% ━━━━╸─────── 23/57 6.4it/s 3.8s<5.3s
[K        4/5      3.22G     0.7313     0.4316    0.01974         90        224: 42% ━━━━━─────── 24/57 6.2it/s 4.0s<5.3s
[K        4/5      3.22G     0.7282     0.4318    0.01962         96        224: 43% ━━━━━─────── 25/57 6.6it/s 4.2s<4.9s
[K        4/5      3.22G     0.7276     0.4307    0.01958         94        224: 45% ━━━━━─────── 26/57 6.2it/s 4.3s<5.0s
[K        4/5      3.22G     0.7228     0.4275    0.01946         97        224: 47% ━━━━━╸────── 27/57 6.6it/s 4.5s<4.6s
[K        4/5      3.22G      0.719      0.423    0.01936         96        224: 49% ━━━━━╸────── 28/57 6.2it/s 4.7s<4.7s
[K        4/5      3.22G     0.7191     0.4224    0.01944         82        224: 50% ━━━━━━────── 29/57 6.4it/s 4.8s<4.4s
[K        4/5      3.22G     0.7181      0.421    0.01944         94        224: 52% ━━━━━━────── 30/57 6.1it/s 5.0s<4.4s
[K        4/5      3.22G      0.717     0.4198    0.01939         94        224: 54% ━━━━━━╸───── 31/57 6.5it/s 5.1s<4.0s
[K        4/5      3.22G     0.7128     0.4176    0.01929         94        224: 56% ━━━━━━╸───── 32/57 6.2it/s 5.3s<4.0s
[K        4/5      3.22G      0.708     0.4164    0.01914         82        224: 57% ━━━━━━╸───── 33/57 6.5it/s 5.4s<3.7s
[K        4/5      3.22G     0.7047     0.4152    0.01907         91        224: 59% ━━━━━━━───── 34/57 6.2it/s 5.6s<3.7s
[K        4/5      3.22G     0.7035     0.4157    0.01905         93        224: 61% ━━━━━━━───── 35/57 6.5it/s 5.8s<3.4s
[K        4/5      3.22G     0.7006     0.4137    0.01895         99        224: 63% ━━━━━━━╸──── 36/57 6.2it/s 6.0s<3.4s
[K        4/5      3.22G     0.6982     0.4118    0.01887        104        224: 64% ━━━━━━━╸──── 37/57 6.6it/s 6.1s<3.0s
[K        4/5      3.22G     0.6947     0.4101    0.01878         91        224: 66% ━━━━━━━━──── 38/57 6.3it/s 6.3s<3.0s
[K        4/5      3.22G      0.691     0.4084     0.0187        104        224: 68% ━━━━━━━━──── 39/57 6.6it/s 6.4s<2.7s
[K        4/5      3.22G     0.6898     0.4079    0.01865         92        224: 70% ━━━━━━━━──── 40/57 6.3it/s 6.6s<2.7s
[K        4/5      3.22G      0.687     0.4066    0.01861         85        224: 71% ━━━━━━━━╸─── 41/57 6.6it/s 6.7s<2.4s
[K        4/5      3.22G      0.685     0.4065    0.01854        105        224: 73% ━━━━━━━━╸─── 42/57 6.2it/s 6.9s<2.4s
[K        4/5      3.22G      0.681     0.4045    0.01842        100        224: 75% ━━━━━━━━━─── 43/57 6.5it/s 7.1s<2.2s
[K        4/5      3.22G     0.6792     0.4028    0.01836        100        224: 77% ━━━━━━━━━─── 44/57 6.2it/s 7.2s<2.1s
[K        4/5      3.22G     0.6761     0.4011    0.01825         96        224: 78% ━━━━━━━━━─── 45/57 6.4it/s 7.4s<1.9s
[K        4/5      3.22G     0.6744     0.4001    0.01825         81        224: 80% ━━━━━━━━━╸── 46/57 6.1it/s 7.6s<1.8s
[K        4/5      3.22G     0.6696     0.3988    0.01814         85        224: 82% ━━━━━━━━━╸── 47/57 6.4it/s 7.7s<1.6s
[K        4/5      3.22G     0.6711     0.3998     0.0182         88        224: 84% ━━━━━━━━━━── 48/57 6.0it/s 7.9s<1.5s
[K        4/5      3.22G     0.6704     0.3997    0.01818         87        224: 85% ━━━━━━━━━━── 49/57 6.4it/s 8.0s<1.2s
[K        4/5      3.22G     0.6713     0.3982    0.01819        107        224: 87% ━━━━━━━━━━╸─ 50/57 6.1it/s 8.2s<1.1s
[K        4/5      3.22G     0.6703      0.398    0.01815         88        224: 89% ━━━━━━━━━━╸─ 51/57 6.4it/s 8.4s<0.9s
[K        4/5      3.22G      0.669     0.3974    0.01812         91        224: 91% ━━━━━━━━━━╸─ 52/57 6.2it/s 8.5s<0.8s
[K        4/5      3.22G     0.6662      0.396    0.01805         99        224: 92% ━━━━━━━━━━━─ 53/57 6.5it/s 8.7s<0.6s
[K        4/5      3.22G     0.6647     0.3953    0.01805         82        224: 94% ━━━━━━━━━━━─ 54/57 6.1it/s 8.9s<0.5s
[K        4/5      3.22G     0.6621     0.3942    0.01795        101        224: 96% ━━━━━━━━━━━╸ 55/57 6.5it/s 9.0s<0.3s
[K        4/5      3.22G     0.6598     0.3927    0.01786         14        224: 98% ━━━━━━━━━━━╸ 56/57 7.0it/s 9.1s<0.1s
[K        4/5      3.22G     0.6598     0.3927    0.01786         14        224: 100% ━━━━━━━━━━━━ 57/57 6.2it/s 9.1s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 25% ━━━───────── 1/4 1.5it/s 0.2s<2.0s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 50% ━━━━━━────── 2/4 2.6it/s 0.4s<0.8s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 75% ━━━━━━━━━─── 3/4 3.3it/s 0.6s<0.3s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 6.1it/s 0.7s
all        204        204       0.97      0.985      0.993      0.834
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
[K        5/5      3.22G     0.5713     0.3701    0.01454        101        224: 0% ──────────── 0/57  0.1s
[K        5/5      3.22G     0.5744     0.3566    0.01499        100        224: 1% ──────────── 1/57 1.7it/s 0.3s<33.6s
[K        5/5      3.22G     0.6176     0.3408    0.01627         91        224: 3% ──────────── 2/57 3.3it/s 0.5s<16.4s
[K        5/5      3.22G       0.63     0.3462    0.01672         90        224: 5% ╸─────────── 3/57 3.9it/s 0.6s<13.7s
[K        5/5      3.22G     0.6178      0.339    0.01622        102        224: 7% ╸─────────── 4/57 4.8it/s 0.8s<11.1s
[K        5/5      3.22G     0.6196     0.3464    0.01632         97        224: 8% ━─────────── 5/57 5.0it/s 1.0s<10.3s
[K        5/5      3.22G     0.6074     0.3454    0.01601         90        224: 10% ━─────────── 6/57 5.7it/s 1.1s<9.0s
[K        5/5      3.22G     0.6071     0.3558    0.01613         86        224: 12% ━─────────── 7/57 5.6it/s 1.3s<8.9s
[K        5/5      3.22G      0.608     0.3487    0.01608         97        224: 14% ━╸────────── 8/57 6.0it/s 1.4s<8.2s
[K        5/5      3.22G     0.6062     0.3474    0.01603         98        224: 15% ━╸────────── 9/57 5.7it/s 1.6s<8.3s
[K        5/5      3.22G     0.5997     0.3507    0.01582         88        224: 17% ━━────────── 10/57 6.2it/s 1.8s<7.5s
[K        5/5      3.22G     0.5962     0.3492    0.01583         93        224: 19% ━━────────── 11/57 6.0it/s 2.0s<7.7s
[K        5/5      3.22G     0.5977     0.3456    0.01572         97        224: 21% ━━╸───────── 12/57 6.3it/s 2.1s<7.1s
[K        5/5      3.22G     0.5968     0.3467    0.01576         97        224: 22% ━━╸───────── 13/57 6.0it/s 2.3s<7.3s
[K        5/5      3.22G      0.595     0.3565    0.01563         98        224: 24% ━━╸───────── 14/57 6.4it/s 2.4s<6.7s
[K        5/5      3.22G      0.599     0.3557    0.01581        100        224: 26% ━━━───────── 15/57 6.1it/s 2.6s<6.9s
[K        5/5      3.22G     0.5984     0.3603     0.0157        103        224: 28% ━━━───────── 16/57 6.3it/s 2.8s<6.5s
[K        5/5      3.22G      0.596     0.3627    0.01569         83        224: 29% ━━━╸──────── 17/57 6.0it/s 2.9s<6.6s
[K        5/5      3.22G     0.5943     0.3592    0.01571         90        224: 31% ━━━╸──────── 18/57 6.4it/s 3.1s<6.1s
[K        5/5      3.22G     0.5878     0.3566    0.01554         92        224: 33% ━━━━──────── 19/57 6.1it/s 3.3s<6.2s
[K        5/5      3.22G     0.5833      0.354    0.01539         95        224: 35% ━━━━──────── 20/57 6.4it/s 3.4s<5.8s
[K        5/5      3.22G     0.5851     0.3528    0.01552         88        224: 36% ━━━━──────── 21/57 6.1it/s 3.6s<5.9s
[K        5/5      3.22G     0.5778     0.3488    0.01533         94        224: 38% ━━━━╸─────── 22/57 6.1it/s 3.7s<5.7s
[K        5/5      3.22G     0.5727     0.3504    0.01522         88        224: 40% ━━━━╸─────── 23/57 6.0it/s 3.9s<5.7s
[K        5/5      3.22G     0.5739     0.3521    0.01526        101        224: 42% ━━━━━─────── 24/57 6.5it/s 4.1s<5.1s
[K        5/5      3.22G     0.5714     0.3498    0.01519        100        224: 43% ━━━━━─────── 25/57 6.1it/s 4.2s<5.2s
[K        5/5      3.22G     0.5689     0.3478    0.01513         94        224: 45% ━━━━━─────── 26/57 6.5it/s 4.4s<4.8s
[K        5/5      3.22G     0.5702     0.3518    0.01514         92        224: 47% ━━━━━╸────── 27/57 6.2it/s 4.6s<4.8s
[K        5/5      3.22G     0.5724     0.3533    0.01523         93        224: 49% ━━━━━╸────── 28/57 6.5it/s 4.7s<4.4s
[K        5/5      3.22G     0.5715     0.3524    0.01518        100        224: 50% ━━━━━━────── 29/57 6.2it/s 4.9s<4.5s
[K        5/5      3.22G     0.5706     0.3522    0.01514         99        224: 52% ━━━━━━────── 30/57 6.5it/s 5.0s<4.2s
[K        5/5      3.22G      0.569     0.3508    0.01507         97        224: 54% ━━━━━━╸───── 31/57 6.1it/s 5.2s<4.2s
[K        5/5      3.22G     0.5695     0.3515    0.01504        103        224: 56% ━━━━━━╸───── 32/57 6.4it/s 5.4s<3.9s
[K        5/5      3.22G     0.5671     0.3517    0.01497         96        224: 57% ━━━━━━╸───── 33/57 6.1it/s 5.5s<4.0s
[K        5/5      3.22G     0.5625     0.3484    0.01484         97        224: 59% ━━━━━━━───── 34/57 6.4it/s 5.7s<3.6s
[K        5/5      3.22G     0.5597     0.3493    0.01476         91        224: 61% ━━━━━━━───── 35/57 6.0it/s 5.9s<3.6s
[K        5/5      3.22G     0.5587      0.351    0.01476         93        224: 63% ━━━━━━━╸──── 36/57 6.4it/s 6.0s<3.3s
[K        5/5      3.22G     0.5558      0.352    0.01465        105        224: 64% ━━━━━━━╸──── 37/57 6.1it/s 6.2s<3.3s
[K        5/5      3.22G     0.5534     0.3498    0.01461        100        224: 66% ━━━━━━━━──── 38/57 6.4it/s 6.3s<3.0s
[K        5/5      3.22G     0.5504     0.3478    0.01452         94        224: 68% ━━━━━━━━──── 39/57 6.2it/s 6.5s<2.9s
[K        5/5      3.22G     0.5491     0.3487    0.01451         88        224: 70% ━━━━━━━━──── 40/57 6.5it/s 6.7s<2.6s
[K        5/5      3.22G      0.548     0.3474    0.01447        100        224: 71% ━━━━━━━━╸─── 41/57 6.1it/s 6.8s<2.6s
[K        5/5      3.22G     0.5473     0.3475    0.01443        104        224: 73% ━━━━━━━━╸─── 42/57 6.5it/s 7.0s<2.3s
[K        5/5      3.22G     0.5448      0.347    0.01437         89        224: 75% ━━━━━━━━━─── 43/57 6.2it/s 7.2s<2.3s
[K        5/5      3.22G     0.5423     0.3454    0.01431         92        224: 77% ━━━━━━━━━─── 44/57 6.5it/s 7.3s<2.0s
[K        5/5      3.22G     0.5422      0.345    0.01431         91        224: 78% ━━━━━━━━━─── 45/57 6.1it/s 7.5s<2.0s
[K        5/5      3.22G     0.5393     0.3438    0.01425         82        224: 80% ━━━━━━━━━╸── 46/57 6.4it/s 7.6s<1.7s
[K        5/5      3.22G     0.5375     0.3435     0.0142         84        224: 82% ━━━━━━━━━╸── 47/57 6.2it/s 7.8s<1.6s
[K        5/5      3.22G      0.536     0.3421    0.01416        101        224: 84% ━━━━━━━━━━── 48/57 6.4it/s 7.9s<1.4s
[K        5/5      3.22G     0.5343     0.3399    0.01411         86        224: 85% ━━━━━━━━━━── 49/57 6.2it/s 8.1s<1.3s
[K        5/5      3.22G     0.5324     0.3384    0.01406         93        224: 87% ━━━━━━━━━━╸─ 50/57 6.5it/s 8.3s<1.1s
[K        5/5      3.22G     0.5316     0.3369    0.01403         95        224: 89% ━━━━━━━━━━╸─ 51/57 6.2it/s 8.4s<1.0s
[K        5/5      3.22G     0.5298     0.3366    0.01398         86        224: 91% ━━━━━━━━━━╸─ 52/57 6.6it/s 8.6s<0.8s
[K        5/5      3.22G     0.5309     0.3377    0.01402         83        224: 92% ━━━━━━━━━━━─ 53/57 6.2it/s 8.8s<0.6s
[K        5/5      3.22G     0.5303     0.3368    0.01401         95        224: 94% ━━━━━━━━━━━─ 54/57 6.5it/s 8.9s<0.5s
[K        5/5      3.22G     0.5293     0.3371    0.01399         91        224: 96% ━━━━━━━━━━━╸ 55/57 6.2it/s 9.1s<0.3s
[K        5/5      3.22G     0.5301     0.3346    0.01399         12        224: 100% ━━━━━━━━━━━━ 57/57 6.2it/s 9.2s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 25% ━━━───────── 1/4 1.5it/s 0.2s<2.0s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 50% ━━━━━━────── 2/4 2.6it/s 0.4s<0.8s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 75% ━━━━━━━━━─── 3/4 3.4it/s 0.6s<0.3s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 6.2it/s 0.6s
all        204        204      0.961      0.959      0.988      0.884
5 epochs completed in 0.019 hours.
Optimizer stripped from /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26_train/yolo26_leaf/weights/last.pt, 44.0MB
Optimizer stripped from /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26_train/yolo26_leaf/weights/best.pt, 44.0MB
Validating /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26_train/yolo26_leaf/weights/best.pt...
Ultralytics 8.4.75 🚀 Python-3.14.6 torch-2.12.1+cu130 CUDA:0 (NVIDIA GeForce RTX 5060 Laptop GPU, 8151MiB)
YOLO26m summary (fused): 132 layers, 20,350,223 parameters, 0 gradients, 67.8 GFLOPs
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 25% ━━━───────── 1/4 1.2it/s 0.2s<2.5s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 50% ━━━━━━────── 2/4 1.9it/s 0.5s<1.1s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 75% ━━━━━━━━━─── 3/4 2.4it/s 0.8s<0.4s
[K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 4/4 4.3it/s 0.9s
all        204        204      0.961       0.96      0.988      0.883
Speed: 0.1ms preprocess, 1.1ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to [1m/mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26_train/yolo26_leaf[0m
[+] YOLOv26 leaf detector saved successfully to /mnt/c/Users/Swapnil/Projects/Leaf_Disease_Detection/models/yolo26_leaf_detector.pt
Completed successfully.

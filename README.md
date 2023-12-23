# MLOps project


## !!!Задание доделал до дедлайна, см предыдущие коммиты

This is a simple MLOps project.

The ML part itself is meant to be as simple as possible in order to concentrate more on the MLOps part.

## ML part:

* a simple sklearn model is trained on the iris dataset
* then inferred on a separate part of the dataset

## MLOps part:

#### Дерево
.
├── Dockerfile
├── README.md
├── client.py
├── data
│   ├── prediction_target.csv
│   ├── test_data.csv
│   ├── test_data.csv.dvc
│   ├── test_target.csv
│   ├── test_target.csv.dvc
│   ├── train_data.csv
│   ├── train_data.csv.dvc
│   ├── train_target.csv
│   └── train_target.csv.dvc
├── docker-compose.yaml
├── iris_classifiers
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   └── utils.cpython-310.pyc
│   ├── infer.py
│   ├── train.py
│   └── utils.py
├── model_repository
│   └── sklearn-onnx
│       └── 1
│           ├── config.pbtxt
│           ├── model.onnx
│           └── model.onnx.dvc
├── poetry.lock
├── pyproject.toml
└── setup.cfg


#### Скрипт для конвертации лежит в iris_classifiers/utils

#### Система
Ubuntu 22.04.3 LTS

#### CPU info

Число ядер: 32

Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         40 bits physical, 48 bits virtual
  Byte Order:            Little Endian
CPU(s):                  32
  On-line CPU(s) list:   0-31
Vendor ID:               GenuineIntel
  Model name:            Intel Xeon Processor (Cascadelake
                         )
    CPU family:          6
    Model:               85
    Thread(s) per core:  2
    Core(s) per socket:  8
    Socket(s):           2
    Stepping:            6
    BogoMIPS:            4200.00

$free
               total        used        free      shared  buff/cache   available
Mem:       198056300     3555804    12905864      278060   181594632   192766052
Swap:         114964         776      114188

-------

#### Оптимизация

До оптимизации
Concurrency: 8, throughput: 12589.1 infer/sec, latency 634 usec

После оптимизации:
Увеличение кол-ва инстансов модели


count: 2
Concurrency: 8, throughput: 10601.2 infer/sec, latency 753 usec

count: 4
Concurrency: 8, throughput: 13008.6 infer/sec, latency 613 usec

count: 10
Concurrency: 8, throughput: 11801.4 infer/sec, latency 676 usec


max_queue_delay_microseconds: 1000
Concurrency: 8, throughput: 11398.7 infer/sec, latency 700 usec

max_queue_delay_microseconds: 100
Concurrency: 8, throughput: 10462.3 infer/sec, latency 763 usec

max_queue_delay_microseconds: 500
Concurrency: 8, throughput: 12710.2 infer/sec, latency 628 usec

max_queue_delay_microseconds: 2000
Concurrency: 8, throughput: 12385.3 infer/sec, latency 644 usec

max_queue_delay_microseconds: 3000
Concurrency: 8, throughput: 12428.8 infer/sec, latency 642 usec

max_queue_delay_microseconds: 4000
Concurrency: 8, throughput: 10913.2 infer/sec, latency 731 usec



WINNER

instance_group [
    {
        count: 8
        kind: KIND_CPU
    }
]

dynamic_batching: { max_queue_delay_microseconds: 500 }


Concurrency: 8, throughput: 13291.6 infer/sec, latency 600 usec

При увеличении количества инстансов модели растет параллелизм, что ведет к улучшению общего показателя производительности. Однако, это требует больше вычислительных ресурсов.

max_queue_delay_microseconds: Увеличение этого параметра может улучшить общую throughput, позволяет более долго задерживать запросы в очереди перед обработкой. Но слишком большая задержка может увеличить latency для каждого запроса.

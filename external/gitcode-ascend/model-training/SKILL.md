---
name: external-gitcode-ascend-model-training
description: Model training on Ascend NPU. Invoke when user wants to launch training
  script and monitor training progress.
original-name: model-training
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# Model Training Skill

Launch and monitor model training on Ascend NPU environment.

## When to Invoke

- User wants to start model training
- User wants to run performance training
- User wants to run accuracy training
- User asks about training options

## Information to Collect

Ask user for the following:

```
1. Training mode preference:
   - Performance training (FP32/FP16)
   - Accuracy training (full epochs)
   - Custom training
   
2. Number of GPUs/NPUs to use

3. Batch size (if custom)

4. Any specific training parameters
```

## Training Options

Provide user with training options:

```
Training Options:
1. Performance Training (FP32, 8 GPUs) - Quick performance test
2. Performance Training (FP16, 8 GPUs) - Quick performance test with mixed precision
3. Accuracy Training (24 epochs) - Full accuracy training
4. Custom training configuration
```

## Workflow

### Step 1: Select Training Mode

Ask user to select training mode or provide custom configuration.

### Step 2: Prepare Environment

Ensure environment variables are set:

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=$(hostname -I | awk '{print $1}')
export HCCL_CONNECT_TIMEOUT=1200
```

### Step 3: Launch Training

Execute training script based on selected mode.

### Step 4: Monitor Progress

Check training log and verify training started successfully.

## BEVFormer Training Commands

### Performance Training (FP32, 8 GPUs)

```bash
cd <working_directory>
bash test/train_performance_8p_base_fp32.sh --batch-size=1 --num-npu=8
```

### Performance Training (FP16, 8 GPUs)

```bash
cd <working_directory>
bash test/train_performance_8p_base_fp16.sh --batch-size=1 --num-npu=8
```

### Accuracy Training (24 epochs)

```bash
cd <working_directory>
bash test/train_full_8p.sh --batch-size=1
```

### Custom Training

```bash
cd <model_directory>
bash ./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py <num_gpus>
```

## Training Verification

### Check Training Started

```bash
# Check training process
ps aux | grep torchrun

# Check training log
tail -20 <working_directory>/test/output/train_performance_8p_base_fp32.log
```

### Success Indicators

Training is considered successfully started when:

1. Log file shows training iterations
2. Loss values are being printed
3. No error messages in recent log
4. Process is running (torchrun)

Example successful log:
```
2026-03-12 09:47:40,838 - mmdet - INFO - Epoch [1][26/41]       lr: 7.333e-05, eta: 0:36:35, time: 1.883, data_time: 0.023, memory: 25332, loss_cls: 1.2614, loss_bbox: 1.7827
```

## Monitoring Training

### Check Training Log

```bash
tail -f <working_directory>/test/output/train_performance_8p_base_fp32.log
```

### Key Metrics

- **Loss values**: loss_cls, loss_bbox, total loss
- **Learning rate**: lr
- **Time per iteration**: time
- **Memory usage**: memory
- **ETA**: estimated time remaining

### Validation Metrics (per epoch)

- **NDS**: NuScenes Detection Score
- **mAP**: mean Average Precision
- **mATE**: mean Translation Error
- **mASE**: mean Scale Error
- **mAOE**: mean Orientation Error

## Training Completion

Training is complete when:

1. All epochs finished
2. Final validation metrics printed
3. Checkpoint saved

Check for completion:
```bash
grep "Saving checkpoint" <log_file> | tail -1
```

## Troubleshooting

### Training Fails to Start

1. Check NPU devices: `npu-smi info`
2. Check environment variables
3. Check conda environment is activated
4. Check dataset and weights are linked

### Out of Memory

1. Reduce batch size
2. Reduce number of GPUs
3. Enable gradient checkpointing

### Training Hangs

1. Check NPU status: `npu-smi info`
2. Check for dead processes: `ps aux | grep python`
3. Check log for last activity

### Port Already in Use

```bash
# Kill existing training processes
pkill -f torchrun
pkill -f dist_train
```

## Log File Locations

| Training Mode | Log File |
|---------------|----------|
| FP32 Performance | `test/output/train_performance_8p_base_fp32.log` |
| FP16 Performance | `test/output/train_performance_8p_base_fp16.log` |
| Accuracy | `test/output/train_full_8p.log` |

## Reference

- Training scripts: `<model_directory>/test/`
- Model config: `<model_directory>/projects/configs/`
- DrivingSDK README: `DrivingSDK/model_examples/<model>/README.md`

# Stage2 批量生成脚本

这个脚本用于从 7 个 stage2 类别的**最佳 checkpoint**生成图像与 mask，并按 `类别/image` 与 `类别/mask` 保存。

## 默认输出目录

```
/root/autodl-tmp/model/ArSDM_exps/generated/<class>/image
/root/autodl-tmp/model/ArSDM_exps/generated/<class>/mask
```

## 默认行为

- 每个类别生成 1500 张
- 使用配置文件中的 `exp.exp_name` 来定位 `ArSDM_exps` 下的最新训练目录
- 优先使用 `checkpoints/` 下的**非 last.ckpt**（若存在），否则使用 `last.ckpt`

## 运行方式

```bash
/root/autodl-tmp/environment/skin/bin/python /root/autodl-tmp/ArSDM-main/generate_stage2_samples.py
```

## 常用参数

- `--num_images`: 每类生成数量（默认 1500）
- `--batch_size`: 采样 batch size（默认 8）
- `--ddim_steps`: 采样步数（默认 20）
- `--output_root`: 输出目录根路径
- `--configs`: 指定哪些配置文件

示例：

```bash
/root/autodl-tmp/environment/skin/bin/python /root/autodl-tmp/ArSDM-main/generate_stage2_samples.py \
  --num_images 1500 \
  --batch_size 8 \
  --ddim_steps 20 \
  --output_root /root/autodl-tmp/model/ArSDM_exps/generated
```

#!/usr/bin/env bash
ABLATION_GROUPS=(A B C D E F G H I)
declare -A GROUP_ROOT=(
  [A]="${GROUP_A_ROOT:-/root/autodl-tmp/output/ablation_complete/A}"
  [B]="${GROUP_B_ROOT:-/root/autodl-tmp/output/ablation_complete/B}"
  [C]="${GROUP_C_ROOT:-/root/autodl-tmp/output/ablation_complete/C}"
  [D]="${GROUP_D_ROOT:-/root/autodl-tmp/output/ablation_complete/D}"
  [E]="${GROUP_E_ROOT:-/root/autodl-tmp/output/ablation_complete/E}"
  [F]="${GROUP_F_ROOT:-/root/autodl-tmp/output/ablation_complete/F}"
  [G]="${GROUP_G_ROOT:-/root/autodl-tmp/output/ablation_complete/G}"
  [H]="${GROUP_H_ROOT:-/root/autodl-tmp/output/ablation_complete/H}"
  [I]="${GROUP_I_ROOT:-/root/autodl-tmp/output/ablation_complete/I}"
)
CLASSES=(akiec bcc bkl df mel nv vasc)
declare -A EXPECTED=([akiec]=1500 [bcc]=1500 [bkl]=500 [df]=1500 [mel]=500 [nv]=500 [vasc]=1500)

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=transfuser_agent \
experiment_name=william_training_transfuser_agent \
scene_filter=all_scenes \
split=mini \
scene_filter.frame_interval=1 \
trainer.params.max_epochs=20000 \
trainer.params.check_val_every_n_epoch=100 \
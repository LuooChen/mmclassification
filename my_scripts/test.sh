#!/bin/bash
start_time=`date '+%Y-%m-%d %H:%M:%S'`
left_brace="["
right_brace="]"
start_train_str="${left_brace}${start_time}${right_brace} train start"
echo $start_train_str >> /home/superdisk/pedestrian-fine-recognition/my_scripts/crontab.log
end_time=`date '+%Y-%m-%d %H:%M:%S'`
end_train_str="${left_brace}${end_time}${right_brace} train end"
echo $end_train_str >> /home/superdisk/pedestrian-fine-recognition/my_scripts/crontab.log
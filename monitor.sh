
while true
do
  # 改动项1 用来查看你前一个在执行的训练是否还存在，这里为(train.py)
  count=$(ps -ef | grep python | grep -c train.py)	# 
  echo "$count train.py are running."
  if [ True ]  # 改动项2， 根据之前被占用的显卡数调整
    then
     # 改动项3 查询第1块gpu的容量2p 第2块3p  第3块4p  第四块5p 依次类推
     # 这里只需要检测0,1GPU的显存情况便可以知道上一个程序是否执行完
     # 这里的11代指gpustat第11个单元的内容，需要自己进行更改！！！每个人不一定一样，例如:
     
     # 我在8卡集群上就为9，则命令为 gpustat | awk '{print $9}' | sed -n '2p'才能打印出当前显存
     # 我在本地单卡机器上为11，则命令为 gpustat | awk '{print $11}' | sed -n '2p'才能打印出当前显存
     stat1=$(gpustat | awk '{print $11}' | sed -n '2p')	# 2p代表 GPU 0	
     stat2=$(gpustat | awk '{print $11}' | sed -n '3p') # 3p代表 GPU 1
     stat3=$(gpustat | awk '{print $11}' | sed -n '4p')
     stat4=$(gpustat | awk '{print $11}' | sed -n '5p')
     stat_arr=($stat1 $stat2 $stat3 $stat4)
     echo '当前显存:'${stat_arr[0]}'M, '${stat_arr[1]}'M, '${stat_arr[2]}'M, '${stat_arr[3]}'M'
     gpu_available=0
     gpu_available_index_arr=()
     # 得到空闲GPU的数量和对应的序号
     for i in ${!stat_arr[@]}
     do
       # 如果显存占用小于100M，继续
       if [ "${stat_arr[$i]}" -lt 5000 ]
       then
         gpu_available=$[gpu_available+1]
         gpu_available_index_arr[${#gpu_available_index_arr[@]}]=$i
       fi
     done
     echo '-可用GPU数:'$gpu_available', 第'${gpu_available_index_arr[@]}'块GPU可用'
     # 如果GPU数大于指定数量，取指定数量GPU开始训练
     if [ $gpu_available -ge 4 ]
     then
       echo 'start training.'
       # 需要执行的python shell脚本
    #    sh ./run_code.sh
       bash baichuan2.sh
       break # 防止下一次循环又重复运行上一行命令
     fi
  fi
  echo "waiting for training..."
  sleep 300	# 每300s执行一次，对于一般实验来说，每5分钟执行一次即可,可以自行调整
done

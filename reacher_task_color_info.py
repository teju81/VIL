import numpy as np
import random

def find_contrasting_colors(color_list, color_pallete, color_dist = 100):
    done = False
    while not done:
        done = True
        color = random.sample(color_pallete,1)[0]
        for i in range(len(color_list)):
            ref_color = color_list[i]
            if np.sqrt(np.sum(np.square(np.subtract(color,ref_color)))) < color_dist:
                done = False
                break
        if done:
            color_list.append(color)

    return color_list

def gen_reacher_task_color_info(num_objects = 4, filename = "task_color_info.txt", color_dist = 100):
	color_pallete = [(c1,c2,c3) for c1 in range(0,255,20) for c2 in range(0,255,20) for c3 in range(0,255,20)]
	with open(filename, 'w') as filehandle:
		for i in range(1800):
			color_list = []
			for j in range(num_objects):
				color_list = find_contrasting_colors(color_list, color_pallete, color_dist)
			for j in range(num_objects):
				for  k in range(3):
					filehandle.write(str(color_list[j][k]) + ' ')
			filehandle.write('\n')
	filehandle.close()


def get_reacher_task_color_info(num_objects = 4, filename = "task_color_info.txt", task_color_index = 3):
	filename = "task_color_info.txt"
	num_objects = 4
	with open(filename, 'r') as filehandle:  
		current_line = 0  
		for line in filehandle:
			if current_line == task_color_index:
				color_list = line
				break
			current_line += 1
		color_list = color_list.strip('\n').split(' ')
		color_list = [(int(color_list[i]), int(color_list[i+1]), int(color_list[i+2])) for i in range(0,len(color_list)-2,3)]
		filehandle.close()
	return color_list
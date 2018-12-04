# -*- coding: utf-8 -*-
"""
    Created on Wed Aug 19 15:25:47 2015
    
    @author: alexbewley
    
    @detail: This is a modified version of an earlier script used for generating a caffe network prototxt file.
    See Usage below and Zongyuan Ge's mixture DCNN paper for more details.
    
"""
import sys

class_list = []
class_list.append("class")
class_list.append("subclass")
class_list.append("color")
class_list.append("sunroof")
class_list.append("luggage_carrier")
class_list.append("open_cargo_area")
class_list.append("enclosed_cab")
class_list.append("spare_wheel")
class_list.append("wrecked")
class_list.append("flatbed")
class_list.append("ladder")
class_list.append("enclosed_box")
class_list.append("soft_shell_box")
class_list.append("harnessed_to_a_cart")
class_list.append("ac_vents")

if(len(sys.argv) < 4):
  print("Usage:\n$ python " + sys.argv[0] + " base-model.prototxt N K\n")
  print("Where:\n - base-model is the prototxt with special TAGS")
  print(" - N is the number of class outputs")
  print(" - K is the number of subset expert networks to replicate from the base model")
  exit()

base_model_definition = sys.argv[1]
num_outputs = int(sys.argv[2])
num_experts = int(sys.argv[3])

for s in range(num_experts):
  with open(base_model_definition,'r') as fin:
    for i,line in enumerate(fin):
      new_line = line.replace('EXPERT_NUM','se%d'%(s+1))
      new_line = new_line.replace('NUM_OUTPUTS',str(num_outputs))
      print(new_line),
    print(" ")

for s in range(num_experts):
  print("layer {\n  type: \"Slice\"\n  name:\"slice-fc8-se%d\"\n  bottom:\"fc8-se%d\""%(s+1,s+1))
  for i in range(num_outputs):
    print("  top: \"slice%d-%d\""%(s+1,i+1))
  print("}")
  
  print("layer {\n  type: \"Eltwise\"\n  name:\"max-fc8-se%d\"\n  top:\"max-fc8-se%d\""%(s+1,s+1))
  for i in range(num_outputs):
    print("  bottom: \"slice%d-%d\""%(s+1,i+1))
  print("  eltwise_param{\n    operation:MAX\n  }\n}")


print("layer {\n  name: \"concat\"\n")
for s in range(num_experts):
  print("  bottom: \"max-fc8-se%d\""%(s+1))
print("  top: \"conf-ss\"\n  type: \"Concat\"\n  concat_param {\n    concat_dim: 1\n  }\n}")


print("layer {\n  name: \"prob-ss\"\n  type: \"Softmax\"\n  bottom: \"conf-ss\"\n  top: \"prob-ss\"\n}")


print("layer {\n  name: \"slice-prob-ss\"\n  type: \"Slice\"\n  bottom: \"prob-ss\"")
for s in range(num_experts):
  print("  top: \"prob-sw%d\""%(s+1))
print("}")


for s in range(num_experts):
  print("layer {")
  print("  name: \"repmat-sww%d\""%(s+1))
  print("  type: \"InnerProduct\"\n  bottom: \"prob-sw%d\""%(s+1))
  print("  top: \"repmat-sww%d\""%(s+1))
  print("  param {\n lr_mult: 0\n  decay_mult: 0\n }")
  print("  param {\n lr_mult: 0\n  decay_mult: 0\n }\n inner_product_param {")
  #print("  blobs_lr: 0\n  blobs_lr: 0")
  #print("  weight_decay: 0\n  weight_decay: 0\n  inner_product_param {")
  print("    num_output: %d"%(num_outputs))
  print("    weight_filler {\n      type: \"constant\"\n      value: 1\n    }\n    bias_filler {\n      type: \"constant\"\n      value: 0\n    }\n  }\n}")

  print("layer {\n    type: \"Eltwise\"")
  print("  name: \"weighted-prob-ss%d\""%(s+1))
  print("  bottom: \"fc8-se%d\""%(s+1))
  print("  bottom: \"repmat-sww%d\""%(s+1))
  print("  top: \"weighted-prob-ss%d\""%(s+1))
  print("  eltwise_param {\n    operation: PROD\n  }\n}")


for j in range(num_outputs):
	print("layer {\n  name: \"sum-weighted-prob%d\"\n  type: \"Eltwise\"" %(j+1))
	for s in range(num_experts):
	  print("  bottom: \"weighted-prob-ss%d\"" %(s+1))
	print("  top: \"prob-%d\"" %(j+1))
	print("  eltwise_param {\n    operation: SUM\n  }\n}")

	#print accuracy layer
	print("layer {\n" + \
	"  name: \"accuracy-%d\"\n" %(j+1) + \
	"  type: \"Accuracy\"\n" + \
	"  bottom: \"prob-%d\"\n" %(j+1) + \
	"  bottom: \"%s\"\n" %(class_list[j])+ \
	"  top: \"accuracy-%d\"\n" %(j+1) + \
	"  include: { phase: TEST }\n}")

	#print loss layer
	print("layer {\n" + \
	"  bottom: \"prob-%d\" \n" %(j+1) +\
	"  bottom: \"%s\"\n" %(class_list[j]) + \
	"  top: \"loss-%d\"\n" %(j+1) + \
	"  name: \"loss-%d\"\n" %(j+1) + \
	"  type: \"SoftmaxWithLoss\"\n" +\
	"  loss_weight: 1\n}")


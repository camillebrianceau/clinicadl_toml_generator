import streamlit as st
import json as jsn
import pandas as pd

from to_toml import dumps
from pathlib import Path


import os




class Form():
	def __init__(self):
		st.title(":octopus: ClinicaDL - TOML generator")
		st.header("Define a network task from TOML file.")
		st.divider()
		st.write( "Since the train pipeline has a many options, the command line " 
	       "can be long and difficult to use. To avoid this we created the `--config_file` "
		   "option that allows the user to give a configuration file with all the options "
		   "they need to the command line. The command line will then first load the default "
		   "values, then overwrite the loaded values with the one specified in the "
		   "configuration file before running the job. ")
		st.write("[TOML format](https://toml.io/en/) is a human readable format, thus it is "
	   		"easy to write a configuration file with any text editor. The user just needs to "
			"specify the value of the option in front of the option name in the file. ")
		
		st.write("If you need more information about an argument you can check ClinicaDL documentation [here](https://clinicadl.readthedocs.io/en/latest/Train/Introduction/)")
		st.sidebar.markdown("# Your TOML")

		self.toml_dict = {
			"Model": {},
			"Architecture": {},
			"Classification": {},
			"Regression": {},
			"Reconstruction": {},
			"Computational": {},
			"Reproducibility": {},
			"Transfer_learning": {},
			"Mode": {},
			"Data": {},
			"Cross_validation": {},
			"Optimization": {},
		}

		

	
	def create_text_input(self, label_:str, help_:str= None, space:bool=True, value_:str = ""):
		st.markdown(label_, help = help_)
		value = st.text_input("", label_visibility = "collapsed", key = label_, value = value_)
		if space:
			st.text("")
		return value 
	
	def create_number_input(self, label_:str, min_=0, val_= 0,help_:str=None, space:bool=True):
		st.markdown(label_, help = help_)
		value = st.number_input("",min_value=min_, value=val_, label_visibility = "collapsed", key = label_)
		if space:
			st.text("")
		return value 
	
	def create_checkbox(self, label_, help_=None, value = False):
		value = st.checkbox(label_, help = help_, value = value)
		st.text("")
		return value 
	
	def create_multiselect(self, label_, list_, help_ = None, val_ : list = None, space= False):
		st.markdown(label_, help = help_)
		value = st.multiselect("", list_ , label_visibility = "collapsed", default = val_, key = label_)
		if space : 
			st.text("")
		return value 

	def create_select_box(self, label_:str, list_: list, help_:str, space:bool=True, index_:int = 0, value= None):
		st.markdown(label_, help = help_)
		value = st.selectbox("", list_ , label_visibility = "collapsed", key = label_,index = index_)
		if space : 
			st.text("")
		return value 

	def create_new_section(self, label: str):
		st.divider()
		st.subheader(label)
		st.divider()
		st.sidebar.markdown(f"## {label}")


	def write_dict(self, section: str, name: str, value):
		self.toml_dict[section].update({name: value})

	def create_page(self):
		#============================GENERAL INFORMATION================================
		self.create_new_section("General information")
		
		help_multi_cohort = "It is possible to use ClinicaDL's functions on several datasets at the same time."
		lbl_multi_cohort = "Check this box if you want to do **multi cohort** training"
		multi_cohort = self.create_checkbox(label_=lbl_multi_cohort, help_ = help_multi_cohort)
		st.sidebar.markdown(f"**multi-cohort**: :blue[{multi_cohort}]")

		if multi_cohort :
			help_caps_multi = (" TSV file must contain two sts:\n"
							"- `cohort` the name of the cohort (must correspond to the values in `TSV_DIRECTORY`),\n"
							"- `path` the path to the corresponding [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.")

			caps_dir = self.create_text_input("Write the path to an input TSV file containing path to CAPS directory:", help_caps_multi, False)
			if not(caps_dir.endswith(".tsv")):
				if caps_dir not in ["", None, "."]:
					st.warning(f"{caps_dir} is not a TSV file.")
			st.text("")
			st.sidebar.markdown(f"**caps_directory**: :green[{caps_dir}]")
		else :
			caps_dir = self.create_text_input("Write the path to the input folder containing the neuroimaging data in a [CAPS](https://aramislab.paris.inria.fr/clinica/docs/public/latest/CAPS/Introduction/) hierarchy.", None, True)
			# caps_dir = Path(caps_dir)
			# if not(caps_dir.is_dir()):
			# 	if caps_dir not in ["", None, "."]:
			# 		st.warning(f"{caps_dir} is not a directory.")
			# st.text("")
			st.sidebar.markdown(f"**caps_directory**: :green[{caps_dir}]")

		help_json="The preprocessing json file stored in the `CAPS_DIRECTORY` that corresponds to the `clinicadl prepare-data` output. This will be used to load the correct tensor inputs with the wanted preprocessing."
		
		# if not multi_cohort :
		# 	if caps_dir.is_dir():
		# 		list_json = []
		# 		list_json.append("Chose the preprocessing json file.")
		# 		list_json2 = [i for i in caps_dir.iterdir() if i.suffix == ".json"]
		# 		list_json = list_json + list_json2
		# 		list_json.append("Enter other path.")
		# 		preprocessing_json = self.create_select_box("Give the name of the preprocessing json.", list_json, help_json,)
				
		# 		if preprocessing_json!= "Chose the preprocessing json file.":
		# 			if preprocessing_json == "Enter other path.":
		# 				preprocessing_json2 = self.create_text_input("Enter the path to the json file:")
						
		# 			if Path(preprocessing_json).is_file():
		# 				with st.expander(f"{preprocessing_json}", expanded= True) :
		# 					with preprocessing_json.open(mode="r") as f:
		# 							parameters_json = jsn.load(f)
		# 					st.json(parameters_json)
		# 			else:
		# 				if preprocessing_json != "":	
		# 					st.warning(f"The file {preprocessing_json} doesn't exist.")
		# 		self.preprocessing_name= Path(preprocessing_json).name
		# 		st.sidebar.markdown(f"**preprocessing_json**: {self.preprocessing_name}" )
		# st.text("")
		# st.text("")
			
		preprocessing_json = self.create_text_input("Enter the name of the json file.",)

		tsv_dir = self.create_text_input("Give the input folder of a TSV file tree generated by `clinicadl tsvtools {split|kfold}`.", None, False)
		# kfold_json = Path(tsv_dir) / "kfold.json"
		# split_json = Path(tsv_dir) / "split.json"
		# if kfold_json.is_file():
		# 	with st.expander(f"{kfold_json}", expanded= True): 
		# 		with kfold_json.open(mode="r") as f:
		# 			par_kfold_json = jsn.load(f)
		# 		st.json(par_kfold_json)
		# if split_json.is_file():
		# 	with st.expander(f"{split_json}", expanded= True):
		# 		with split_json.open(mode="r") as f:
		# 			par_split_json = jsn.load(f)
		# 		st.json(par_split_json)
		st.text("")
		st.sidebar.markdown(f"**tsv_directory**: {tsv_dir}" )

		lbl_maps_dir =("Give a path to a new folder where the results will be stored")
		maps_dir = self.create_text_input(lbl_maps_dir, None, True)
		st.sidebar.markdown(f"**maps_directory**: {maps_dir}" )
		
		network_task = self.create_select_box('Choose the type of task learnt by the network.', ('Classification','Reconstruction', 'Regression'), "", True)
		st.sidebar.markdown(f"**network_task**: {network_task}" )

		#====================================MODEL=====================================
		section_ = "Model"
		st.divider()
		st.subheader(f"Model and architecture for {network_task}")
		st.divider()
		st.sidebar.markdown(f"## {section_}")
			
		classif_list = ("resnet18","Conv4_FC3", "Conv5_FC3", "ResNet3D", "SqueezeExcitationCNN", "Stride_Conv5_FC3", "Other")
		reg_list = ("Conv4_FC3", "Conv5_FC3", "ResNet3D", "Stride_Conv5_FC3", "Other")
		recons_list = ("AE_Conv4_FC3", "AE_Conv5_FC3", "CAE_half", "CVAE_3D", "CVAE_3D_final_conv", "CVAE_3D_half", "Vanilla3DdenseVAE", "Vanilla3DspacialVAE", "VanillaDenseVAE", "VanillaSpatialVAE", "Other")

		if network_task == "Classification":
			st.write("The objective of the classification is to attribute a class to input images. The criterion loss is the cross entropy between the ground truth and the network output. The evaluation metrics are the accuracy, sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV) and balanced accuracy (BA).")
			architecture1 = self.create_select_box("Chose your architecture",classif_list, help_= "The network must output a vector of length equals to the number of classes")
			architecture = architecture1

		elif network_task == "Regression":
			st.write("The objective of the regression is to learn the value of a continuous variable given an image. The criterion loss is the mean squared error between the ground truth and the network output. The evaluation metrics are the mean squared error (MSE) and mean absolute error (MAE).")
			architecture1 = self.create_select_box("Choose your architecture.", reg_list, help_= "The network has only one output node.")
			architecture = architecture1

		elif network_task == "Reconstruction":
			st.write("The objective of the reconstruction is to learn to reconstruct images given in input. The criterion loss is the mean squared error between the input and the network output. The evaluation metrics are the mean squared error (MSE) and mean absolute error (MAE).")
			architecture1 = self.create_select_box("Choose your architecture.", recons_list, help_ = "The network outputs an image of the same size as the input.")
			architecture = architecture1

		if architecture1 == "Other":
			architecture2 = self.create_text_input("If you had added a new model which doesn't appear in the list, please enter the name of the network class here:")
			architecture = architecture2
		
		st.sidebar.markdown(f"**architecture**: {architecture}" )
		self.write_dict(section_, "architecture", architecture)
		
		
		multi_network = self.create_checkbox("Check this box if you want to do **multi network** training.", None)
		st.sidebar.markdown(f"**multi_network**: {multi_network}" )
		self.write_dict(section_, "multi_network", multi_network)


		#=================================ARCHITECTURE=================================
		section_ = "Architecture"
		st.sidebar.markdown(f"## {section_}")

		dropout = self.create_number_input("Chose the rate of dropout applied in dropout layers.", min_=0, val_ =0)
		st.sidebar.markdown(f"**dropout**: {dropout}")
		self.write_dict(section_, "dropout", dropout)

		if network_task == "Classification":
			section_ = "Classification"

			label = self.create_text_input("Chose the name of the column containing the label for the classification task.", " It must be a categorical variable, but may be of any type", value_="diagnosis")
			st.sidebar.write(f"**label**: {label}")
			self.write_dict(section_, "label", label)

			st.write("Chose metrics used to select networks according to the best validation performance.")
			selection_metric = st.multiselect("", ("accuracy", "sensitivity", "specificity", "PPV", "NPV", "BA", "loss"), default="loss", label_visibility="collapsed")
			st.sidebar.write(f"**selection_metric**: {selection_metric}")
			self.write_dict(section_, "slection_metric", selection_metric)

			if multi_network :
				selection_threshold = self.create_number_input("Chose a selection threshold used for soft-voting.", 0.0, 0.0, "It is only taken into account if several images are extracted from the same original 3D image (i.e. `num_networks > 1`).")
				st.sidebar.write(f"selection_threshold: {selection_threshold}")
				self.write_dict(section_, "selection_threshold", selection_threshold)
			
			loss = self.create_select_box("Chose the name of the loss used to optimize the classification task.", ("CrossEntropyLoss", "MultiMarginLoss"), "Must correspond to a Pytorch class.", value = "CrossEntropyLoss")
			st.sidebar.write(f"**loss**: {loss}")
			self.write_dict(section_, "loss", loss)
			


		elif network_task == "Regression":
			section_ = "Regression"

			label = self.create_text_input("Chose the name of the column containing the label for the classification task.", "It must be a continuous variable (float or int)", value_="age")
			st.sidebar.write(f"**label**: {label}")
			self.write_dict(section_, "label", label)

			st.write("Chose metrics used to select networks according to the best validation performance.")
			selection_metric = st.multiselect("", ("MSE", "MAE", "loss"), default="loss", label_visibility="collapsed")
			st.sidebar.write(f"**selection_metric**: {selection_metric}")
			self.write_dict(section_, "slection_metric", selection_metric)

			loss = self.create_select_box("Chose the name of the loss used to optimize the regression task.", ("L1Loss", "MSELoss", "KLDivLoss", "BCEWithLogitsLoss", "HuberLoss", "SmoothL1Loss"), "Must correspond to a Pytorch class.", value="MSELoss")
			st.sidebar.write(f"**loss**: {loss}")
			self.write_dict(section_, "loss", loss)
			
		elif network_task == "Reconstruction":
			section_ = "Reconstruction"

			st.write("Chose metrics used to select networks according to the best validation performance.")
			selection_metric = st.multiselect("", ("MSE", "MAE", "PSNR", "SSIM", "loss"), default="loss", label_visibility="collapsed")
			st.sidebar.write(f"**selection_metric**: {selection_metric}")
			self.write_dict(section_, "slection_metric", selection_metric)

			loss = self.create_select_box("Chose the name of the loss used to optimize the reconstruction task.", ( "L1Loss", "MSELoss", "KLDivLoss", "BCEWithLogitsLoss", "HuberLoss", "SmoothL1Loss", "VAEGaussianLoss", "VAEBernoulliLoss", "VAEContinuousBernoulliLoss",), "Must correspond to a Pytorch class.", value="MSELoss")
			st.sidebar.write(f"**loss**: {loss}")
			self.write_dict(section_, "loss", loss)
			
		#=================================COMPUTATIONAL================================
		section_ = "Computational"
		self.create_new_section("Computational")

		gpu = self.create_checkbox("Uncheck the box if you don't want to use a GPU acceleration.", "Default behavior is to try to use a GPU and to raise an error if it is not found.", True)
		st.sidebar.write(f"**gpu**: {gpu}")
		self.write_dict(section_, "gpu", gpu)

		n_proc = self.create_number_input("Chose the number of workers used by the DataLoader.", min_ = 1, val_ = 2)
		st.sidebar.write(f"**n_proc**: {n_proc}")
		self.write_dict(section_, "n_proc", n_proc)

		batch_size = self.create_number_input("Chose the size of the batch used in the DataLoader.", min_= 1, val_ = 8)
		st.sidebar.write(f"**batch_size**: {batch_size}")
		self.write_dict(section_, "batch_size", batch_size)

		evaluation_steps = self.create_number_input( "Gives the number of iterations to perform an [evaluation internal to an epoch](Details.md#evaluation).", help_= "Default will only perform an evaluation at the end of each epoch", min_ = 0, val_ = 0)
		st.sidebar.write(f"**evaluation_steps**: {evaluation_steps}")
		self.write_dict(section_, "evaluation_steps", evaluation_steps)

		#=================================REPRODUCIBILITY==============================
		section_ = "Reproducibility"
		self.create_new_section("Reproducibility")

		seed = self.create_number_input("Chose the value used to set the seed of all random operations.", min_ = 0, val_ = 0, help_ = "Default samples a seed and uses it for the experiment.")
		st.sidebar.write(f"**seed**: {seed}")
		self.write_dict(section_, "seed", seed)
			
		deterministic = self.create_checkbox("Check to force the training process to be deterministic.", help_=" If any non-deterministic behaviour is encountered will raise a RuntimeError")
		st.sidebar.write(f"**deterministic**: {deterministic}")
		self.write_dict(section_, "deterministic", deterministic)

		if deterministic:
			compensation = self.create_select_box("Chose how CUDA will compensate to obtain a deterministic behaviour.", list_= ("memory", "time"), help_= "The computation time will be longer, or the computations will require more memory space.", value= "memory")
			st.sidebar.write(f"**compensation**: {compensation}")
			self.write_dict(section_, "compensation", compensation)
		
		#===============================TRANSFER LEARNING==============================
		section_ = "Transfer_learning"
		self.create_new_section("Transfer Learning")

		with st.expander("Expand to chose option for transfer learning"):
			transfer_path = self.create_text_input("Chose the path to the model used for transfer learning.")
			st.sidebar.write(f"**transfer_path**: {transfer_path}")
			self.write_dict(section_, "transfer_path", transfer_path)

			transfer_selection_metric = self.create_text_input("Chose the transfer learning selection metric. ",("loss"))
			st.sidebar.write(f"**transfer_selection_metric**: {transfer_selection_metric}")
			self.write_dict(section_, "transfer_selection_metric", transfer_selection_metric)

		#====================================MODE======================================
		section_ = "Mode"
		self.create_new_section("Mode")

		use_extracted_features = self.create_checkbox("use extracted features")
		st.sidebar.write(f"**use_extracted_features**: {use_extracted_features}")
		self.write_dict(section_, "use_extracted_features", use_extracted_features)

		#====================================DATA======================================
		section_ = "Data"
		self.create_new_section("Data")

		self.diagnoses = st.multiselect("diagnoses",("AD", "CN"))

		baseline = self.create_checkbox("Check to load only _baseline.tsv files instead of .tsv files comprising all the sessions")
		st.sidebar.write(f"**baseline**: {baseline}")
		self.write_dict(section_, "baseline", baseline)

		normalize = self.create_checkbox("Check to disable min-max normalization that is performed by default", value= True)
		st.sidebar.write(f"**normalize**: {normalize}")
		self.write_dict(section_, "normalize", normalize)

		data_augmentation = self.create_multiselect("Chose the list of data augmentation transforms applied to the training data", ("Noise", "Erasing", "CropPad", "Smoothing", "Motion", "Ghosting", "Spike", "BiasField", "RandomBlur", "RandomSwap"), val_ = None)
		st.sidebar.write(f"**data_augmentation**: {data_augmentation}")
		self.write_dict(section_, "data_augmentation", data_augmentation)

		sampler = self.create_select_box("Chose the sampler used on the training set. It must be chosen in",("random", "weighted"), help_="weighted will give a stronger weight to underrepresented classes")
		st.sidebar.write(f"**sampler**: {sampler}")
		self.write_dict(section_, "sampler", sampler)

		size_reduction=st.checkbox("size reduction")
		st.sidebar.write(f"**size_reduction**: {size_reduction}")
		self.write_dict(section_, "size_reduction", size_reduction)

		if size_reduction:
			size_reduction_factor=st.number_input("size reduction factor", value = 2)
			st.sidebar.write(f"**size_reduction_factor**: {size_reduction_factor}")
			self.write_dict(section_, "size_reduction_factor", size_reduction_factor)

		#==============================CROSS VALIDATION===============================
		section_ = "Cross_validation"
		self.create_new_section("Cross validation")

		n_splits = self.create_number_input("Chose a number of splits k to load in the case of a k-fold cross-validation.", min_ = 0, val_ = 0, help_="Default will load a single-split")
		st.sidebar.write(f"**n_splits**: {n_splits}")
		self.write_dict(section_, "n_splits", n_splits)

		split_list = []
		for i in range(n_splits):
			split_list.append(f"{i}")
		split_selected = self.create_multiselect("Chose a subset of folds that will be used for training.", split_list, val_=split_list, help_="By default all splits available are used.")
		st.sidebar.write(f"**split**: {split_selected}")
		self.write_dict(section_, "split", split_selected)
		#=================================OPTIMIZATION================================
		section_ = "Optimization"
		self.create_new_section("Optimization")

		optimizer = self.create_select_box("Chose the name of the optimizer used to train the network.", ("Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "ASGD", "NAdam", "RAdam", "RMSprop", "SGD"), help_=None ,index_=2 )
		st.sidebar.write(f"**optimizer**: {optimizer}")
		self.write_dict(section_, "optimizer", optimizer)

		help_epochs = "If early stopping is disabled, or if its stopping criterion was never reached, training stops when the maximum number of epochs epochs is reached."
		epochs = self.create_number_input("Chose the maximum number of epochs", val_ = 20, min_ = 1, help_ = help_epochs)
		st.sidebar.write(f"**epochs**: {epochs}")
		self.write_dict(section_, "epochs", epochs)

		#st.number_input(label, min_value=None, max_value=None, value=, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible")
		learning_rate =  self.create_text_input("Enter the learning rate used to perform weight update.", value_ = "0.0001",space=False)
		if float(learning_rate)<= 0.000000:
			st.warning("Learning rate must be a positive number.")
		elif float(learning_rate)>=1:
			st.warning("Learning rate must be < 1.")
		st.text("")
		st.sidebar.write(f"**learning_rate**: {float(learning_rate)}")
		self.write_dict(section_, "learning_rate", float(learning_rate))

		weight_decay = self.create_text_input("Chose the weight decay used by the optimizer", value_ = "0.0001", space = False)
		if float(weight_decay)<= 0.000000:
			st.warning("Weight decay must be a positive number.")
		elif float(weight_decay)>=1:
			st.warning("Weight decay must be < 1.")
		st.text("")
		st.sidebar.write(f"**weight_decay**: {float(weight_decay)}")
		self.write_dict(section_, "weight_decay", float(weight_decay))

		patience = self.create_number_input("Chose the number of epochs for early stopping patience.", val_= 0, min_ = 0)
		st.sidebar.write(f"**patience**: {patience}")
		self.write_dict(section_, "patience", patience)

		tolerance = self.create_text_input("Chose the value used for early stopping tolerance.", value_ = "0.0", space = False)
		if float(tolerance)< 0.000000:
			st.warning("tolerance must be a positive number.")
		elif float(tolerance)>=1:
			st.warning("tolerance must be < 1.")
		st.text("")
		st.sidebar.write(f"**tolerance**: {float(tolerance)}")
		self.write_dict(section_, "tolerance", float(tolerance))

		accumulation_steps = self.create_number_input("Chose the number of iterations during which gradients are accumulated before performing the weights update.", val_ = 1, min_ = 0,help_="This allows to virtually increase the size of the batch.")
		st.sidebar.write(f"**accumulation_steps**: {accumulation_steps}")
		self.write_dict(section_, "accumulation_steps", accumulation_steps)

		profiler = self.create_checkbox("Check this box if you want to add a profiler.",)
		st.sidebar.write(f"**profiler**: {profiler}")
		self.write_dict(section_, "profiler", profiler)

		#=================================LOAD YOUR TOML================================
		st.divider()
		st.subheader("Load you TOML and run clinicadl")

		output_toml = dumps(self.toml_dict)

		st.write("The command line to launch the training is :")
		st.markdown(f"`clinicadl train {network_task} {caps_dir} {preprocessing_json} {tsv_dir} {maps_dir} --config_file <path to the download toml file>`")
		st.download_button(
			label="Download your TOML",
			data=output_toml,
			file_name='config.toml',
		)


	
			






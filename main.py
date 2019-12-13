import classification as cl


if __name__ == "__main__":
	md = cl.model("sample_data/titanic.csv")
	df = md.csv_as_df()


	columns = md.column_names()
	columns = columns[0].split("\t")

	y_name = "Survived"
	X_names = [x for x in columns if x !=y_name]
	X = md.set_X(X_names)
	y = md.set_Y(y_name)
	train_X, test_X, train_y, test_y = md.test_train(X,y, 0.3)









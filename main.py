import classification as cl


if __name__ == "__main__":
	md = cl.model("titanic.csv")
	md = md.csv_as_df()


	# columns = md.column_names()
	# columns = columns[0].split("\t")
	# y = md.set_Y("Survived")








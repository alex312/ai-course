class T():
    def __init__(self,name):
    		pass

class Team(T):
    A= "A"
	def __init__(self):  #There are two underlines on each side
		self.name=name  #The must assignment

	def ReportName(self):
		print(self.name)


AlphaTeam=Team('Sarah')
AlphaTeam.ReportName()
print(Team.A)


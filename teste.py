
class Quadrado(object):
	def __init__(self,lado):
		self.lado = lado
	def MudaValorLados(self,novo_lado):
		self.lado = novo_lado
	def RetornaValorDados(self):
		return self.lado
	def CalculaArea(self):
		return self.lado**2

class Retangulo(object):
	def __init__(self,largura,altura):
		self.altura = altura
		self.largura = largura
	def MudaValorLados(self,nova_altura,nova_largura):
		self.altura = nova_altura
		self.largura = nova_largura
	def RetornaValorLados(self):
		return self.altura, self.largura
	def CalculaArea(self):
		return self.altura*self.largura
	def CalculaPerimetro(self):
		return self.altura * 2 + self.largura * 2


x = float(input("Informe o comprimento do local: "))
y = float(input("Informe a altura do local: "))
sala = Retangulo(x,y)
def programa_usuario1(sala,x,y):
	area_sala = sala.CalculaArea()

	piso = Quadrado(30.4)
	rodape = Retangulo(30.4,15.2)
	area_piso = piso.CalculaArea()
	area_rodape = rodape.CalculaArea()

	numero_pisos = area_sala/area_piso
	numero_rodape = area_sala/area_rodape

	return numero_pisos,numero_rodape

class Televisor(object):
	def __init__(self,canal,volume,):
		self.menorCanal = 2
		self.maiorCanal = 100
		self.maiorVolume = 80
		if self.menorCanal > canal and self.maiorCanal < canal:
			self.canal = canal
		else:
			print("Escolha canal valido")
	def RetornaDados(self):
		return "canal: "self.canal,"volume: "self.volume
t = Televisor(1,1)
print t.RetornaDados











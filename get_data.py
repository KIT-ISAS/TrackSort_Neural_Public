import os

for name in ['Zylinder', 'Pfeffer', 'Kugeln', 'Weizen']:
  # download data
  url = 'pollithy.com/{}.zip'.format(name)
  zip_ = '{}.zip'.format(name)
  os.system("wget -N " + url)
  # unzip data
  os.system("unzip -n " + zip_)
  os.system("mv " + name + " data/" + name)
  os.system("rm " + zip_)

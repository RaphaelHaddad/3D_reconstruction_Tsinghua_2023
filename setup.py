from distutils.core import setup
setup(
  name = 'kp_imc2023',         # How you named your package folder (MyLib)
  packages = ['kp_imc2023'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Package for the Image Matching Challenge 2023',   # Give a short description about your library
  author = 'Kevin Pruvost \ Raphael El Haddad \ Borislav Pavlov',                   # Type in your name
  author_email = 'pruvostkevin0@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/RaphaelHaddad/3D_reconstruction_Tsinghua_2023',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['CV', 'IMC'],   # Keywords that define your package best
  install_requires=open("requirements.txt", "r").read().split("\n"),
  classifiers=[
    'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.9',
  ],
)
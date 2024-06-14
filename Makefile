main.out: main.cpp json.hpp Makefile instances
	g++ -DNDEBUG -std=c++1z -Wall -Wextra -Ofast -march=native -pthread main.cpp -s -o main.out

json.hpp:
	curl -fLO https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp

instances:
	mkdir -p instances
	curl -fL 'http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/bqp{50,100,250,500,1000}.txt' -o 'instances/bqp#1.txt'
	curl -fL 'http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/bqp2500.gz' | gzip -c -d > instances/bqp2500.txt
	curl -fL 'https://web.stanford.edu/~yyye/yyye/Gset/G[1-54]' -o 'instances/G#1'
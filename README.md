# Лабораторная работа №1
## Реализация метода обратного распространения ошибки

### Запуск программы
 Для запуска решения необходимо установить Pythone и библиотеку Numpy.

 Запуск программы производится из командной строки командой следующего вида:

 python BackPropagation.py -n 1000 -t 200 -s 150 -l 0,02 -e 20
 
 Рсшифровка параметров:
 
 -n - размер обучающей выборки
 
 -t - размер тестовой выборки
 
 -s - количество нейронов на скрытом слое
 
 -l - скорость обучения

 -e - количество эпох
 
 В случае, если какие-либо из аргументов не будут заданы, соответствующим переменным будут присвоены значения по умолчанию.
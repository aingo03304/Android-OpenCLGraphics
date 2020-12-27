cmake_minimum_required(VERSION 2.6)

add_library(
	Csparse

	STATIC

	src/csparse.c)
target_include_directories(Csparse PRIVATE "include")

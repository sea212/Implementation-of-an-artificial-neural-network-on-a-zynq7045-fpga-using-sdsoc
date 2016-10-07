################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/main.cpp \
../src/neural_net.cpp 

OBJS += \
./src/main.o \
./src/neural_net.o 

CPP_DEPS += \
./src/main.d \
./src/neural_net.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: SDS++ Compiler'
	sds++ -Wall -O0 -g -I"../src" -c -fmessage-length=0 -MT"$@" -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<" -sds-hw neural_net neural_net.cpp  -clkid 2 -sds-end -sds-pf zc706
	@echo 'Finished building: $<'
	@echo ' '



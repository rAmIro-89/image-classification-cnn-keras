# cnn
git pull(traer cambios de la misma rama)
git pull origin main(traer cambios del main de origen aka github)
git checkout _____(cambiar de rama a otra rama)
git checkout -b ____(cambiar de rama a nueva nueva, el ___ representa el nuevo nombre. IMPORTANTE, usar este comando desde main)
git add .(agrega todos los cambios que hiciste, p.ej: creaste un nuevo file)
git commit -am "nombre de la rama y que estas subiendo"(sirve para registra tus cambios)
git push(sube tus cambios a github)


Este trabajo practico es de la materia de Procesamiento del Aprendizaje Automatico, esta hecho por Zoe Mlinarevic Medl y Ramiro Ottone.

Para agilizar el trabajo en equipo decidimos hacer el desarrollo en un ide local como vscode y subir los cambios a un gestor de versiones como github.

En las etapas iniciales del proyecto la cnn se encuentra con los siguientes datos: "accuracy: 0.6171 - loss: 1.0872", segun los objetivos del trabajo practico hay que aumentar la accuracy y bajar la perdida mediante la alteracion de los hiperparametros. Una accuracy ideal segun lo charlado en clase seria una que ronde entre el 80 y el 85%.

Mediante un incremento del hiperparametro de epochs los datos son lso siguientes: "18ms/step - accuracy: 0.7617 - loss: 0.6879", al haber corrido con los mismos datos, hay un riesgo de overfitting si seguimos por la misma ruta de incremento de epochs, para confirmar que los resultados son correctos, debemos hacer una validacion(TODO)
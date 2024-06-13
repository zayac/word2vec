package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type ModelBuilder struct {
	layers []*Layer
	inputs int
}

func NewModelBuilder(inputs int) *ModelBuilder {
	return &ModelBuilder{inputs: inputs}
}

func (m *ModelBuilder) AddLayer(outputs int, act Activator) *ModelBuilder {
	if len(m.layers) == 0 {
		m.layers = append(m.layers, NewLayer(
			randomDense(m.inputs, outputs),
			randomDense(1, outputs),
			act))
	} else {
		m.layers = append(m.layers, NewLayer(
			randomDense(m.layers[len(m.layers)-1].Len(), outputs),
			mat.NewDense(1, outputs, nil),
			act))
	}
	return m
}

func (m ModelBuilder) Build() []*Layer { return m.layers }

func randomDense(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = 0.1 - 0.2*rand.Float64()
	}
	return mat.NewDense(rows, cols, data)
}

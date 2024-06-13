package main

import (
	"fmt"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/gonum/mat"
)

const (
	datasetPath     = "dataset"
	tokenWindow     = 2
	embeddingVector = 10
	screenWidth     = 640
	screenHeight    = 480
)

func main() {
	ds, err := newDataset(datasetPath)
	if err != nil {
		log.Fatal(err)
	}
	vocabSize := len(ds.IDToWord)
	model := NewModelBuilder(vocabSize).
		AddLayer(embeddingVector, LinearActivator{}).
		AddLayer(vocabSize, SoftmaxActivator{}).
		Build()
	app := NewGame(screenWidth, screenHeight)
	epochStart := 0
	go func() {
		for _, d := range []struct {
			t      *Trainer
			epochs int
		}{
			{NewTrainer(5e-2), 45},
			{NewTrainer(1e-2), 1000},
		} {
			d.t.Train(model, ds, tokenWindow, epochStart, epochStart+d.epochs, app.RenderLoss)
			epochStart += d.epochs
		}

		var y mat.Dense
		in := NewInferencer(model)
		defer in.Done()
		predict := func(w string) string {
			vocabSize := len(ds.IDToWord)
			idx, ok := ds.WordToID[w]
			if !ok {
				panic(fmt.Sprintf("word %q not found in dataset", w))
			}
			x := hotVec(idx, vocabSize)
			in.Infer(mat.NewDense(1, vocabSize, x))
			in.Result(&y)
			return ds.IDToWord[indexOfMax(y.RowView(0).(*mat.VecDense))]
		}
		fmt.Println(predict("machine"))
	}()
	if err := ebiten.RunGame(app); err != nil {
		panic(err)
	}
}

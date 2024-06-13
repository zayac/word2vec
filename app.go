package main

import (
	"image"
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

func plotImg(ps ...plot.Plotter) *image.RGBA {
	p := plot.New()
	p.Add(append([]plot.Plotter{
		plotter.NewGrid(),
	}, ps...)...)
	img := image.NewRGBA(image.Rect(0, 0, screenWidth, screenHeight))
	c := vgimg.NewWith(vgimg.UseImage(img))
	p.Draw(draw.New(c))
	return c.Image().(*image.RGBA)
}

type Game struct {
	width, height int
	lossImgCh     chan *image.RGBA
	lossImg       *ebiten.Image
	lossData      plotter.XYs
}

func NewGame(width, height int) *Game {
	canvasImage := ebiten.NewImage(width, height)
	canvasImage.Fill(color.White)
	return &Game{
		lossImgCh: make(chan *image.RGBA, 1),
		width:     width,
		height:    height,
	}
}

func (g *Game) Draw(screen *ebiten.Image) {
	select {
	case img := <-g.lossImgCh:
		g.lossImg = ebiten.NewImageFromImage(img)
	default:
	}
	if g.lossImg != nil {
		screen.DrawImage(g.lossImg, nil)
	}
}

func (g *Game) Layout(width, height int) (screenWidth, screenHeight int) {
	return g.width, g.height
}

func (g *Game) Update() error {
	return nil
}

func (g *Game) RenderLoss(lossIncrement plotter.XY) {
	g.lossData = append(g.lossData, lossIncrement)
	lossLines, _ := plotter.NewLine(g.lossData)
	img := plotImg(lossLines)
	select {
	case <-g.lossImgCh: // Drain the channel.
		g.lossImgCh <- img // Put the new image in.
	case g.lossImgCh <- img: // Or just put the new image in.
	}
}

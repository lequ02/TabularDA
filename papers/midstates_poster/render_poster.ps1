# Exports the PPTX to PDF and a PNG preview with PowerPoint (Windows only).
param([string]$Out = "$PSScriptRoot\..\..\output\pdf\midstates_poster")
$pptx = (Resolve-Path "$PSScriptRoot\ffts_midstates_poster.pptx").Path
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
$app = New-Object -ComObject PowerPoint.Application
$deck = $app.Presentations.Open($pptx, $true, $false, $false)
$deck.SaveAs("$Out\ffts_midstates_poster_48x36.pdf", 32)
$deck.Slides(1).Export("$Out\ffts_midstates_poster_preview.png", "PNG", 1920, 1440)
$deck.Close()
$app.Quit()

param([Parameter(Mandatory=$true)][string]$Scene,
      [Parameter(Mandatory=$true)][string]$Directory)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$pages = Get-Content -LiteralPath $Scene -Raw -Encoding UTF8 | ConvertFrom-Json
function Color([string]$hex) { [System.Drawing.ColorTranslator]::FromHtml($hex) }
foreach ($page in $pages) {
    if ($page.filename -notmatch '^(overview|details_[0-9]+)\.png$') { throw 'Unsafe output name' }
    $target = Join-Path $Directory $page.filename
    if (Test-Path -LiteralPath $target) { throw 'Refusing PNG overwrite' }
    $bitmap = [System.Drawing.Bitmap]::new([int]$page.width,[int]$page.height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $fonts = @{}
    $format = [System.Drawing.StringFormat]::GenericTypographic.Clone()
    $format.FormatFlags = $format.FormatFlags -bor [System.Drawing.StringFormatFlags]::MeasureTrailingSpaces
    try {
        foreach ($item in $page.items) {
            if ($item.op -eq 'rect') {
                $brush = [System.Drawing.SolidBrush]::new((Color $item.fill))
                $pen = [System.Drawing.Pen]::new((Color $item.stroke),1)
                $graphics.FillRectangle($brush,[single]$item.x,[single]$item.y,[single]$item.w,[single]$item.h)
                $graphics.DrawRectangle($pen,[single]$item.x,[single]$item.y,[single]$item.w,[single]$item.h)
                $brush.Dispose(); $pen.Dispose()
            } elseif ($item.op -eq 'text') {
                $key = "$($item.size):$($item.bold)"
                if (-not $fonts.ContainsKey($key)) {
                    $style = if ($item.bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
                    $fonts[$key] = [System.Drawing.Font]::new('Segoe UI',[single]$item.size,$style,[System.Drawing.GraphicsUnit]::Pixel)
                }
                $brush = [System.Drawing.SolidBrush]::new((Color $item.color))
                $graphics.DrawString([string]$item.text,$fonts[$key],$brush,[System.Drawing.PointF]::new([single]$item.x,[single]$item.y),$format)
                $brush.Dispose()
            } elseif ($item.op -eq 'line') {
                $pen = [System.Drawing.Pen]::new((Color $item.color),2)
                if ($item.dashed) { $pen.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash }
                if ($item.arrow) { $pen.CustomEndCap = [System.Drawing.Drawing2D.AdjustableArrowCap]::new(4,5) }
                $points = [System.Drawing.PointF[]]@($item.points | ForEach-Object { [System.Drawing.PointF]::new([single]$_[0],[single]$_[1]) })
                $graphics.DrawLines($pen,$points)
                $pen.Dispose()
            }
        }
        $bitmap.Save($target,[System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        foreach ($font in $fonts.Values) { $font.Dispose() }
        $format.Dispose(); $graphics.Dispose(); $bitmap.Dispose()
    }
}

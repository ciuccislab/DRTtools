X = 1.1*45.0;                  %# A3 paper size
Y = 1.1*35.0;                  %# A3 paper size
xMargin = 0.;               %# left/right margins from page borders
yMargin = 0.;               %# bottom/top margins from page borders
xSize = X - 2*xMargin;     %# figure size on paper (width & height)
ySize = Y - 2*yMargin;     %# figure size on paper (width & height)

%# figure size on screen (50% scaled, but same aspect ratio)
set(gcf, 'Units','centimeters', 'Position',[0 0 xSize ySize]/2)

set(gcf, 'PaperUnits','centimeters')
set(gcf, 'PaperSize',[X Y])
set(gcf, 'PaperPosition',[xMargin yMargin xSize ySize])
set(gcf, 'PaperOrientation','portrait')

;-----------------------------------------
; Initialization routine. Draws the widget
;-----------------------------------------
function hinode_init, reset_state=reset_state
	
	state = {baseWidget: 0L, plotLeftWidget: 0L, plotRightWidget: 0L, frameSlider: 0L, stI: ptr_new(), stV: ptr_new(), xpos: 0, ypos: 0, lambda: ptr_new(),$
		Bpar: ptr_new(), image: 0}

; Read all information
	print, 'Reading Stokes I...'
	restore, '../data/stI_rebinned.idl'
	state.stI = ptr_new(stI)
	
	print, 'Reading Stokes V...'
	restore, '../data/stV_rebinned.idl'
	state.stV = ptr_new(stV)
	
	restore, '../data/wavelengthAxis.idl' 
	state.lambda = ptr_new(lambda)
	
	restore, '../data/Bpar.idl'	
	state.Bpar = ptr_new(Bpar)
	
	xsiz = 512

; Generate the widgets
	state.baseWidget = widget_base(TITLE='Hinode inspector', XSIZE=2.0*xsiz)

	allBase = widget_base(state.baseWidget, /COLUMN, /BASE_ALIGN_CENTER)
	
	horizplotBase = widget_base(allBase, /ROW)
	state.plotLeftWidget = widget_draw(horizplotBase, XSIZE=xsiz, YSIZE=xsiz, /FRAME, UVALUE='CLICK', /MOTION_EVENTS)	
	state.plotRightWidget = widget_draw(horizplotBase, XSIZE=xsiz, YSIZE=xsiz, /FRAME)

	gauge1Base = widget_base(allBase, /ROW)
	
	butbase = widget_base(gauge1Base, /COLUMN, /EXCLUSIVE)
	but1 = widget_button(butBase, UVALUE='CONTINUUM', VALUE='Continuum')
	but2 = widget_button(butBase, UVALUE='FLUXDENSITY', VALUE='Mag. flux density')
	
	widget_control, but1, /SET_BUTTON
	
	widget_control, state.baseWidget, SET_UVALUE=state

	return, state
end

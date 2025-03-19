import React from 'react'
import './Works.css'
import line from '../public/lines.png'
export const Works = () => {
  return (
    <>
    <span className='heading'>How it works</span>
    <div className='work-container'>
        <div className='step'>
            <div className='step-line'>
                <img src={line} width='100%' height='100%'/>
            </div>
            <div className='steplist-cont'>
                <div className='step-list'>
                <span>Go to files and </span>
                <span>upload your video</span>
                </div>
                <div className='step-list'>
                    <span>Analyzing </span>
                    <span>your videos</span>
                </div>
                <div className='step-list'>
                    <span>get your result</span>
                </div>
            </div>
            <div className='step-btn' style={{position:'absolute',top:-42,right:-20,backgroundColor:'#000914'}} >1</div>
            <div className='step-btn' style={{position:'absolute',top:'48%',left:-19,backgroundColor:'#000914'}}>2</div>
            <div className='step-btn' style={{position:'absolute',bottom:-34,right:-20,backgroundColor:'#000914'}}>3</div>
        </div>
        <div className='work-video'>
            <span>Tutorials</span>
        </div>
    </div>
    </>
  )
}

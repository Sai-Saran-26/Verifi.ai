import React from 'react'
import { Navbar } from './Navbar'
import originalimg from '../../public/original.png'
import fakeimg from '../../public/fake.png'
import leftArr from '../../public/leftarr.svg'
import rightArr from '../../public/rightarr.svg'
import './Home.css'

export const Home = () => {
  return (
    <>
    <Navbar />
        <div className='home-container'>
            <div className='home-main'>
                <span>Detect Deepfakes Instantly with</span><br></br>
                <span className='home-main-2'>AI-Powered Precision</span>
            </div>
            <div className='home-sub'>
                <span>Uncover deepfake videos with cutting-edge</span><br></br>
                <span className='home-main-2'>AI detection technology</span>
            </div>
            <div className='home-btn'>
                <button>Try Demo Now</button>
            </div>
            <div className='home-imgs'>
                <span className='Arrow-left'>
                <span className='Arrow-index-1'>original</span> <img src={leftArr} /></span>
                <img src={originalimg} width="190px" height="230px"/>
                <img src={fakeimg} width="200px" height="230px"/>
                <span className='Arrow-right'>
                 <img src={rightArr} /> <span className='Arrow-index-2'>fake</span></span>
            </div>
        </div>

    </>
  )
}
